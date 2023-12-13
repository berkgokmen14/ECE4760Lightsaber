/**
 * Berk Gokmen (bg372), Justin Green (jtg239)
 *
 * This program (adapted from Hunter Adam's FFT demo)
 * changes the LED strip based on surrounding frequency
 * and intensity using an FFT in mode 0. It detects
 * vowels using cepstral analysis in mode 1.
 *
 * HARDWARE CONNECTIONS
 *  - GPIO  5 ---> LED ws2812 pin
 *  - GPIO 15 ---> Mode switch Button
 *  - GPIO 16 ---> VGA Hsync
 *  - GPIO 17 ---> VGA Vsync
 *  - GPIO 18 ---> 330 ohm resistor ---> VGA Red
 *  - GPIO 19 ---> 330 ohm resistor ---> VGA Green
 *  - GPIO 20 ---> 330 ohm resistor ---> VGA Blue
 *  - RP2040 GND ---> VGA GND
 *  - GPIO 26 ---> Audio input [0-3.3V]
 *
 * RESOURCES USED
 *  - PIO state machines 0, 1, and 2 on PIO instance 0
 *  - DMA channels 0, 1, 2, and 3
 *  - ADC channel 0
 *  - 153.6 kBytes of RAM (for pixel color data)
 *
 */

// Include VGA graphics library
#include "vga_graphics.h"
// Include standard libraries
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Include Pico libraries
#include "pico/multicore.h"
#include "pico/stdlib.h"
// Include hardware libraries
#include "hardware/adc.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/pio.h"
#include "ws2812.pio.h"
// Include protothreads
#include "pt_cornell_rp2040_v1.h"

// Define the LED pin
#define LED 25

// Define Button Pin for mode switching
#define BTN 15

// Mode: 0 for disco mode, 1 for vowel detection
int mode = 0;

// === the fixed point macros (16.15) ========================================
typedef signed int fix15;
#define multfix15(a, b) \
    ((fix15)((((signed long long)(a)) * ((signed long long)(b))) >> 15))
#define float2fix15(a) ((fix15)((a)*32768.0))  // 2^15
#define fix2float15(a) ((float)(a) / 32768.0)
#define absfix15(a) abs(a)
#define int2fix15(a) ((fix15)(a << 15))
#define fix2int15(a) ((int)(a >> 15))
#define char2fix15(a) (fix15)(((fix15)(a)) << 15)

/////////////////////////// ADC configuration ////////////////////////////////
// ADC Channel and pin
#define ADC_CHAN 0
#define ADC_PIN 26
// Number of samples per FFT
#define NUM_SAMPLES 256
// Number of samples per FFT, minus 1
#define NUM_SAMPLES_M_1 255
// Length of short (16 bits) minus log2 number of samples (10)
#define SHIFT_AMOUNT 8
// Log2 number of samples
#define LOG2_NUM_SAMPLES 8
// Sample rate (Hz)
#define Fs 8000.0
// ADC clock rate (unmutable!)
#define ADCCLK 48000000.0

#define rgb(r, g, b) (((r) << 5) & RED | ((g) << 2) & GREEN | ((b) << 0) & BLUE)

// DMA channels for sampling ADC (VGA driver uses 0 and 1)
int sample_chan = 2;
int control_chan = 3;

// Detected vowel through cepstral analysis
char vowel = 'a';

// counter to keep track of when to update LED
int update_counter = 0;

// the cepstral filter cutoff qerf
#define cutoff 30

// Max and min macros
#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)

// LED Input Pin
#define WS2812_PIN 5
// Indicates if LED is RGB or RGBW
#define IS_RGBW false

// Size of the array for storing peaks in smoothed power spectrum
#define peak_array_size 4

#define num_leds 71

// Noise threshold for detecting peaks in smoothed power spectrum
fix15 noise = float2fix15(600);

// 0.4 in fixed point (used for alpha max plus beta min)
fix15 zero_point_4 = float2fix15(0.4);

// Here's where we'll have the DMA channel put ADC samples
uint8_t sample_array[NUM_SAMPLES];
// And here's where we'll copy those samples for FFT calculation
fix15 fr[NUM_SAMPLES];
fix15 fi[NUM_SAMPLES];

// Sine table for the FFT calculation
fix15 Sinewave[NUM_SAMPLES];
// Hann window table for FFT calculation
fix15 window[NUM_SAMPLES];

// low pass window for truncating cepstrum
fix15 lp_window[NUM_SAMPLES];

// Pointer to address of start of sample buffer
uint8_t* sample_address_pointer = &sample_array[0];

// Peforms an in-place FFT. Adapted from Hunter Adams FFT demo For more
// information about how this algorithm works, please see
// https://vanhunteradams.com/FFT/FFT.html
void FFTfix(fix15 fr[], fix15 fi[]) {
    unsigned short m;   // one of the indices being swapped
    unsigned short mr;  // the other index being swapped (r for reversed)
    fix15 tr, ti;  // for temporary storage while swapping, and during iteration

    int i,
        j;  // indices being combined in Danielson-Lanczos part of the algorithm
    int L;  // length of the FFT's being combined
    int k;  // used for looking up trig values from sine table

    int istep;  // length of the FFT which results from combining two FFT's

    fix15 wr, wi;  // trigonometric values from lookup table
    fix15 qr, qi;  // temporary variables used during DL part of the algorithm

    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// BIT REVERSAL //////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Bit reversal code below based on that found here:
    // https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
    for (m = 1; m < NUM_SAMPLES_M_1; m++) {
        // swap odd and even bits
        mr = ((m >> 1) & 0x5555) | ((m & 0x5555) << 1);
        // swap consecutive pairs
        mr = ((mr >> 2) & 0x3333) | ((mr & 0x3333) << 2);
        // swap nibbles ...
        mr = ((mr >> 4) & 0x0F0F) | ((mr & 0x0F0F) << 4);
        // swap bytes
        mr = ((mr >> 8) & 0x00FF) | ((mr & 0x00FF) << 8);
        // shift down mr
        mr >>= SHIFT_AMOUNT;
        // don't swap that which has already been swapped
        if (mr <= m) continue;
        // swap the bit-reveresed indices
        tr = fr[m];
        fr[m] = fr[mr];
        fr[mr] = tr;
        ti = fi[m];
        fi[m] = fi[mr];
        fi[mr] = ti;
    }
    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// Danielson-Lanczos //////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Adapted from code by:
    // Tom Roberts 11/8/89 and Malcolm Slaney 12/15/94 malcolm@interval.com
    // Length of the FFT's being combined (starts at 1)
    L = 1;
    // Log2 of number of samples, minus 1
    k = LOG2_NUM_SAMPLES - 1;
    // While the length of the FFT's being combined is less than the number
    // of gathered samples . . .
    while (L < NUM_SAMPLES) {
        // Determine the length of the FFT which will result from combining two
        // FFT's
        istep = L << 1;
        // For each element in the FFT's that are being combined . . .
        for (m = 0; m < L; ++m) {
            // Lookup the trig values for that element
            j = m << k;                          // index of the sine table
            wr = Sinewave[j + NUM_SAMPLES / 4];  // cos(2pi m/N)
            wi = -Sinewave[j];                   // sin(2pi m/N)
            wr >>= 1;                            // divide by two
            wi >>= 1;                            // divide by two
            // i gets the index of one of the FFT elements being combined
            for (i = m; i < NUM_SAMPLES; i += istep) {
                // j gets the index of the FFT element being combined with i
                j = i + L;
                // compute the trig terms (bottom half of the above matrix)
                tr = multfix15(wr, fr[j]) - multfix15(wi, fi[j]);
                ti = multfix15(wr, fi[j]) + multfix15(wi, fr[j]);
                // divide ith index elements by two (top half of above matrix)
                qr = fr[i] >> 1;
                qi = fi[i] >> 1;
                // compute the new values at each index
                fr[j] = qr - tr;
                fi[j] = qi - ti;
                fr[i] = qr + tr;
                fi[i] = qi + ti;
            }
        }
        --k;
        L = istep;
    }
}

// Converts (r, g, b) into a 32 bit RGB representation
static inline uint32_t urgb_u32(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)(r) << 8) | ((uint32_t)(g) << 16) | (uint32_t)(b);
}

// ==================================================
// === convert HSV to rgb value
// ==================================================
// Adapted from Bruce Land's code
// Source:
// https://people.ece.cornell.edu/land/courses/ece4760/RP2040/index_rp2040_MBED.html
int hsv2rgb(float h, float s, float v) {
    float C, X, m, rp, gp, bp;
    unsigned char r, g, b;
    // hsv to rgb conversion from
    // http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
    C = v * s;
    // X = C * (1 - abs((int)(h/60)%2 - 1));
    //  (h/60) mod 2  = (h/60 - (int)(h/60))
    X = C * (1.0 - fabsf(fmodf(h / 60.0, 2.0) - 1.));
    m = v - C;
    if ((0 <= h) && (h < 60)) {
        rp = C;
        gp = X;
        bp = 0;
    } else if ((60 <= h) && (h < 120)) {
        rp = X;
        gp = C;
        bp = 0;
    } else if ((120 <= h) && (h < 180)) {
        rp = 0;
        gp = C;
        bp = X;
    } else if ((180 <= h) && (h < 240)) {
        rp = 0;
        gp = X;
        bp = C;
    } else if ((240 <= h) && (h < 300)) {
        rp = X;
        gp = 0;
        bp = C;
    } else if ((300 <= h) && (h < 360)) {
        rp = C;
        gp = 0;
        bp = X;
    } else {
        rp = 0;
        gp = 0;
        bp = 0;
    }
    // scale to 8-bit rgb
    r = (unsigned char)((rp + m) * 255);
    g = (unsigned char)((gp + m) * 255);
    b = (unsigned char)((bp + m) * 255);
    //

    return urgb_u32(r, g, b);
}

// Converts audio (given intensity and frequency)
// to HSV by mapping frequency to hue and mapping
// intensity to saturation and value, then converts
// HSV to RGB
int audio_to_rgb(fix15 intensity, float f) {
    float sat_and_val = fix2float15(intensity);
    if (sat_and_val > 1) sat_and_val = 1;
    if (sat_and_val < 0.1) sat_and_val = 0.1;
    float intensity_to_saturation = sat_and_val;
    float intensity_to_value = sat_and_val;
    float intensity_to_hue = (f / 1500) * 360;
    return hsv2rgb(intensity_to_hue, intensity_to_saturation,
                   intensity_to_value);
}

// shifts in color to the LED strip
static inline void put_pixel(uint32_t pixel_grb) {
    pio_sm_put_blocking(pio1, 0, pixel_grb << 8u);
}

//====================================
// log2 approx
// see:
// Generation of Products and Quotients Using Approximate Binary Logarithms
// for Digital Filtering Applications,
// IEEE Transactions on Computers 1970 vol.19 Issue No.02
//====================================
// Adapted from Bruce Land's code
// Source:
// https://people.ece.cornell.edu/land/courses/ece4760/RP2040/index_rp2040_MBED.html
void log2_cepstral(fix15 fr[], int length) {
    fix15 log_input, log_output;
    // reduced range variable for interpolation
    fix15 x;
    for (int ii = 0; ii < length; ii++) {
        log_input = fr[ii];
        //
        // check for too small or negative
        // and return smallest log2
        if (log_input <= float2fix15(0.00003)) {
            fr[ii] = int2fix15(-15);
            continue;
        }
        // if the input is less than 2 the scale up by
        // 2^14 so always working on an integer
        // so we can get logs down to input of 0.00003 or so approx -14.85
        int frac_factor = 0;
        if (log_input < int2fix15(2)) {
            // max size of shift to not overflow
            frac_factor = 14;
            log_input <<= frac_factor;
        }

        // temp for finding msb
        fix15 sx;
        sx = log_input;

        // find the most-significant bit
        // equivalent to finding the characteristic of the log
        fix15 y = int2fix15(1);  // value of MSB
        fix15 ly = 0;            // position of MSB
        while (sx > int2fix15(2)) {
            y = y << 1;
            ly = ly + int2fix15(1);
            sx = sx >> 1;
        }
        // bound the bottom and detect negative input values
        // Two-segment approx is good to better than  0.02 log unit
        // equiv to finding the mantissa of the log, then adding the charastic
        // see:
        // Generation of Products and Quotients Using Approximate Binary
        // Logarithms for Digital Filtering Applications, IEEE Transactions on
        // Computers 1970 vol.19 Issue No.02 normalize the bits after dleting
        // MSB
        x = (log_input - y) >> fix2int15(ly);
        fix15 fix_frac_factor = int2fix15(frac_factor);
        // piecewise linear curve fit
        if (x < float2fix15(0.5))
            log_output =
                (ly + multfix15(x, float2fix15(1.163)) + float2fix15(0.0213)) -
                fix_frac_factor;
        else
            log_output =
                (ly + multfix15(x, float2fix15(0.828)) + float2fix15(0.1815)) -
                fix_frac_factor;
        // one segment approx goodd to about 0.07 log unit
        // log_output = (ly + x*0.984 + 0.065) - frac_factor ;
        // and store it
        fr[ii] = log_output;
        //
    }
}

// Computes l1_dist of peaks given f1 and f2
int l1_dist(fix15 peaks[], int f1, int f2) {
    // Compute absolute distance of first component
    fix15 dist1 = int2fix15(f1) - multfix15(peaks[0], float2fix15(31.25));
    fix15 abs_dist1 = absfix15(dist1);

    // Compute absolute distance of second component
    fix15 dist2 = int2fix15(f2) - multfix15(peaks[1], float2fix15(31.25));
    fix15 abs_dist2 = absfix15(dist2);

    // Return sum
    return abs_dist1 + abs_dist2;
}

// Runs on core 0
static PT_THREAD(protothread_fft(struct pt* pt)) {
    // Indicate beginning of thread
    PT_BEGIN(pt);

    printf("Starting capture\n");
    // Start the ADC channel
    dma_start_channel_mask((1u << sample_chan));
    // Start the ADC
    adc_run(true);

    // Declare some static variables
    static int height;          // for scaling display
    static float max_freqency;  // holds max frequency
    static int i;               // incrementing loop variable

    static fix15 max_fr;    // temporary variable for max freq calculation
    static int max_fr_dex;  // index of max frequency

    while (1) {
        // If button is pressed, toggle the mode from 0 to 1 or vice versa
        int updated = 0;

        while (!gpio_get(BTN)) {
            if (!updated) {
                if (mode)
                    mode = 0;
                else
                    mode = 1;
                updated = 1;
            }
        }

        // ==================================================
        // === DISCO MODE
        // ==================================================
        if (mode == 0) {
            // Wait for NUM_SAMPLES samples to be gathered
            // Measure wait time with timer. THIS IS BLOCKING
            dma_channel_wait_for_finish_blocking(sample_chan);

            // Copy/window elements into a fixed-point array
            for (i = 0; i < NUM_SAMPLES; i++) {
                fr[i] = multfix15(int2fix15((int)sample_array[i]), window[i]);
                fi[i] = (fix15)0;
            }

            // Zero max frequency and max frequency index
            max_fr = 0;
            max_fr_dex = 0;

            // Restart the sample channel, now that we have our copy of the
            // samples
            dma_channel_start(control_chan);

            // Compute the FFT
            FFTfix(fr, fi);

            // Find the magnitudes (alpha max plus beta min)
            for (int i = 0; i < (NUM_SAMPLES >> 1); i++) {
                // get the approx magnitude
                fr[i] = abs(fr[i]);
                fi[i] = abs(fi[i]);
                // reuse fr to hold magnitude
                fr[i] = max(fr[i], fi[i]) +
                        multfix15(min(fr[i], fi[i]), zero_point_4);

                // Keep track of maximum
                if (fr[i] > max_fr && i > 4) {
                    max_fr = fr[i];
                    max_fr_dex = i;
                }
            }
            // Compute max frequency in Hz
            max_freqency = max_fr_dex * (Fs / NUM_SAMPLES);

            // Update the LEDs every 5 iterations
            if (update_counter > 5) {
                // Make all num_leds same color
                for (uint i = 0; i < num_leds; ++i) {
                    put_pixel(audio_to_rgb(fr[max_fr_dex], max_freqency));
                }

                update_counter = 0;
            }

            update_counter++;

        }

        // ==================================================
        // === VOWEL RECOGNITION MODE
        // ==================================================
        else {
            // Wait for NUM_SAMPLES samples to be gathered
            // Measure wait time with timer. THIS IS BLOCKING
            dma_channel_wait_for_finish_blocking(sample_chan);

            // preemphasis of higher frequencies
            // 0.95 to 0.97 commonly used
            for (int i = 1; i < NUM_SAMPLES; i++) {
                fr[i] = fr[i] - multfix15(float2fix15(0.95), fr[i - 1]);
            }
            fr[0] = fr[1];

            // Copy/window elements into a fixed-point array
            for (i = 0; i < NUM_SAMPLES; i++) {
                fr[i] = multfix15(int2fix15((int)sample_array[i]), window[i]);
                fi[i] = (fix15)0;
            }

            // Zero max frequency and max frequency index
            max_fr = 0;
            max_fr_dex = 0;

            // Restart the sample channel, now that we have our copy of the
            // samples
            dma_channel_start(control_chan);

            // Compute the FFT
            FFTfix(fr, fi);

            // Find the magnitudes (alpha max plus beta min)
            for (int i = 0; i < (NUM_SAMPLES >> 1); i++) {
                // get the approx magnitude
                fr[i] = abs(fr[i]);
                fi[i] = abs(fi[i]);
                // reuse fr to hold magnitude
                fr[i] = max(fr[i], fi[i]) +
                        multfix15(min(fr[i], fi[i]), zero_point_4);

                // Keep track of maximum
                if (fr[i] > max_fr && i > 4) {
                    max_fr = fr[i];
                    max_fr_dex = i;
                }
            }

            // the log2 of power spectrum
            log2_cepstral(fr, NUM_SAMPLES);

            // the cepstrum (the spectrum of the log(|spectrum|))
            FFTfix(fr, fi);

            // do the cepstral lowpass
            // note that fi[i] is much smaller than fr[i]
            // could probably just zero it
            for (int i = 0; i < NUM_SAMPLES; i++) {
                fr[i] = (multfix15(fr[i], lp_window[i]));
                fi[i] = (multfix15(fi[i], lp_window[i]));
            }

            // convert back to lowpassed spectrum estimate
            FFTfix(fr, fi);

            // Find the magnitudes (alpha max plus beta min)
            for (int i = 0; i < (NUM_SAMPLES >> 1); i++) {
                // get the approx magnitude
                fr[i] = abs(fr[i]);
                fi[i] = abs(fi[i]);
                // reuse fr to hold magnitude
                fr[i] = max(fr[i], fi[i]) +
                        multfix15(min(fr[i], fi[i]), zero_point_4);

                // Keep track of maximum
                if (fr[i] > max_fr && i > 4) {
                    max_fr = fr[i];
                    max_fr_dex = i;
                }
            }

            // Compute max frequency in Hz
            max_freqency = max_fr_dex * (Fs / NUM_SAMPLES);

            fix15 peak_array[peak_array_size];
            fix15 current_pitch_max;
            int peak_num;
            int pitch_max_index;
            // find peaks of points that are over noise threshold for
            // for all 3 points
            peak_num = 0;
            pitch_max_index = 0;
            current_pitch_max = 0;
            for (int i = 3; i < 100; i++) {
                // peak is defined as exceeding the noise threshold and the
                // points declining on the left and right side
                if ((fr[i] > noise) && (fr[i - 1] > noise) &&
                    (fr[i + 1] > noise) && (fr[i - 2] > noise) &&
                    (fr[i + 2] > noise) && (fr[i - 3] > noise) &&
                    (fr[i + 3] > noise) && (fr[i - 1] < fr[i]) &&
                    (fr[i + 1] < fr[i]) && (fr[i - 2] < fr[i - 1]) &&
                    (fr[i + 2] < fr[i + 1]) && (fr[i - 3] < fr[i - 2]) &&
                    (fr[i + 3] < fr[i + 2]) && (peak_num < peak_array_size)) {
                    peak_array[peak_num] = int2fix15(i);
                    peak_num++;
                }
            }

            // L1 distance of ah
            fix15 ah_val = l1_dist(peak_array, 440, 900);

            // L2 distance of ee
            fix15 ee_val = l1_dist(peak_array, 350, 2250);

            // Compute corresponding peaks for frequency 1 and frequency 2
            float peak0 = fix2float15(peak_array[0]) * 31.25;
            float peak1 = fix2float15(peak_array[1]) * 31.25;

            // Catch peaks based on rigid ranges
            if (peak0 < 350 && peak0 > 250)
                vowel = 'e';
            else if (peak0 < 500 && peak0 > 400)
                vowel = 'a';
            // As backup catch peaks based on l1 distance
            else if (ah_val < int2fix15(150) && ah_val < ee_val)
                vowel = 'a';
            else if (ee_val < int2fix15(250) && ee_val < ah_val)
                vowel = 'e';

            // if ah detected, make the lighsaber blue
            if (vowel == 'a') {
                // Update all num_leds to blue
                for (uint i = 0; i < num_leds; ++i) {
                    put_pixel(urgb_u32(0, 0, 255));
                }
            }
            // if ee detected, make the lighsaber red
            else if (vowel == 'e') {
                // Update all num_leds to red
                for (uint i = 0; i < num_leds; ++i) {
                    put_pixel(urgb_u32(255, 0, 0));
                }
            }
        }

        PT_YIELD_usec(1000);
    }
    PT_END(pt);
}

static PT_THREAD(protothread_blink(struct pt* pt)) {
    // Indicate beginning of thread
    PT_BEGIN(pt);
    while (1) {
        // Toggle LED, then wait half a second
        gpio_put(LED, !gpio_get(LED));
        PT_YIELD_usec(500000);
    }
    PT_END(pt);
}

// Core 1 entry point (main() for core 1)
void core1_entry() {
    // Add and schedule threads
    pt_add_thread(protothread_blink);
    pt_schedule_start;
}

// Core 0 entry point
int main() {
    // Initialize stdio
    stdio_init_all();

    // Initialize the VGA screen
    initVGA();

    // Map LED to GPIO port, make it low
    gpio_init(LED);
    gpio_set_dir(LED, GPIO_OUT);
    gpio_put(LED, 0);

    ///////////////////////////////////////////////////////////////////////////////
    // ============================== ADC CONFIGURATION
    // ==========================
    //////////////////////////////////////////////////////////////////////////////
    // Init GPIO for analogue use: hi-Z, no pulls, disable digital input buffer.
    adc_gpio_init(ADC_PIN);

    // Initialize the ADC harware
    // (resets it, enables the clock, spins until the hardware is ready)
    adc_init();

    // Select analog mux input (0...3 are GPIO 26, 27, 28, 29; 4 is temp sensor)
    adc_select_input(ADC_CHAN);

    // Setup the FIFO
    adc_fifo_setup(
        true,   // Write each completed conversion to the sample FIFO
        true,   // Enable DMA data request (DREQ)
        1,      // DREQ (and IRQ) asserted when at least 1 sample present
        false,  // We won't see the ERR bit because of 8 bit reads; disable.
        true    // Shift each sample to 8 bits when pushing to FIFO
    );

    // Divisor of 0 -> full speed. Free-running capture with the divider is
    // equivalent to pressing the ADC_CS_START_ONCE button once per `div + 1`
    // cycles (div not necessarily an integer). Each conversion takes 96
    // cycles, so in general you want a divider of 0 (hold down the button
    // continuously) or > 95 (take samples less frequently than 96 cycle
    // intervals). This is all timed by the 48 MHz ADC clock. This is setup
    // to grab a sample at 10kHz (48Mhz/10kHz - 1)
    adc_set_clkdiv(ADCCLK / Fs);

    // Populate the sine table, Hann window table, and low pass window
    int ii;
    for (ii = 0; ii < NUM_SAMPLES; ii++) {
        Sinewave[ii] =
            float2fix15(sin(6.283 * ((float)ii) / (float)NUM_SAMPLES));
        window[ii] = float2fix15(
            0.5 * (1.0 - cos(6.283 * ((float)ii) / ((float)NUM_SAMPLES))));
        lp_window[ii] = float2fix15(((ii < cutoff) ? float2fix15(100.0) : 0.0));
    }
    lp_window[cutoff + 1] = float2fix15(75.0);
    lp_window[NUM_SAMPLES - cutoff - 1 - 1] = float2fix15(75.0);
    lp_window[cutoff + 2] = float2fix15(50.0);
    lp_window[NUM_SAMPLES - cutoff - 1 - 2] = float2fix15(50.0);
    lp_window[cutoff + 3] = float2fix15(25.0);
    lp_window[NUM_SAMPLES - cutoff - 1 - 3] = float2fix15(25.0);

    // Set GPIO for LED as output

    /////////////////////////////////////////////////////////////////////////////////
    // ============================== ADC DMA CONFIGURATION
    // =========================
    /////////////////////////////////////////////////////////////////////////////////

    // Channel configurations
    dma_channel_config c2 = dma_channel_get_default_config(sample_chan);
    dma_channel_config c3 = dma_channel_get_default_config(control_chan);

    // ADC SAMPLE CHANNEL
    // Reading from constant address, writing to incrementing byte addresses
    channel_config_set_transfer_data_size(&c2, DMA_SIZE_8);
    channel_config_set_read_increment(&c2, false);
    channel_config_set_write_increment(&c2, true);
    // Pace transfers based on availability of ADC samples
    channel_config_set_dreq(&c2, DREQ_ADC);
    // Configure the channel
    dma_channel_configure(sample_chan, &c2,  // channel config
                          sample_array,      // dst
                          &adc_hw->fifo,     // src
                          NUM_SAMPLES,       // transfer count
                          false              // don't start immediately
    );

    // CONTROL CHANNEL
    channel_config_set_transfer_data_size(&c3, DMA_SIZE_32);  // 32-bit txfers
    channel_config_set_read_increment(&c3, false);   // no read incrementing
    channel_config_set_write_increment(&c3, false);  // no write incrementing
    channel_config_set_chain_to(&c3, sample_chan);   // chain to sample chan

    dma_channel_configure(
        control_chan,  // Channel to be configured
        &c3,           // The configuration we just created
        &dma_hw->ch[sample_chan]
             .write_addr,         // Write address (channel 0 read address)
        &sample_address_pointer,  // Read address (POINTER TO AN ADDRESS)
        1,     // Number of transfers, in this case each is 4 byte
        false  // Don't start immediately.
    );

    // Initialize GPIO for set routine button
    gpio_init(BTN);
    gpio_set_dir(BTN, GPIO_IN);
    gpio_pull_up(BTN);

    // LED stuff
    // todo get free sm
    PIO pio = pio1;
    int sm = 0;
    uint offset = pio_add_program(pio, &ws2812_program);

    // Init LED strip pio
    ws2812_program_init(pio, sm, offset, WS2812_PIN, 800000, IS_RGBW);

    // Turn all LEDS off initially and wait 1 second
    for (int x = 0; x < num_leds; x++) {
        put_pixel(0);
    }

    sleep_ms(1000);

    // Launch core 1
    multicore_launch_core1(core1_entry);

    // Add and schedule core 0 threads
    pt_add_thread(protothread_fft);
    pt_schedule_start;
}