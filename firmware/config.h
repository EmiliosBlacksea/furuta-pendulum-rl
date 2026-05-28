#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <math.h>

// --- HARDWARE PINS ---
const int ENCODER_PIN_A = 3;
const int ENCODER_PIN_B = 2;
const int STEP_PIN      = 4;
const int DIR_PIN       = 5;
const int EN_PIN        = 6;
const int LED_PIN       = 13;

// --- PHYSICAL PARAMETERS ---
const float ENCODER_CPR      = 2400.0;
const float ENC_TICKS_TO_RADS = (2.0 * PI) / ENCODER_CPR;
const float STEPS_TO_RADS    = (2.0 * PI) / 3200.0;
const float MAX_VELOCITY     = 3000.0;  // steps/sec

// --- CONTROL TIMING ---
const unsigned long CONTROL_DT_MICROS = 20000;  // 50 Hz (20 ms)

// --- ACTION & VELOCITY FILTERS (must match sim_env.py) ---
const float FILTER_ALPHA = 0.3;   // Action smoothing
const float FILTER_ENC   = 0.35;  // Encoder velocity EMA
const float FILTER_MOT   = 0.4;   // Motor velocity EMA

// --- STATE MACHINE ---
enum ControlMode {
    MODE_STANDALONE,  // Autonomous edge-AI inference on Teensy
    MODE_PC_CONTROL   // PC sends actions; Teensy sends observations (fine-tuning)
};

enum RobotState {
    STATE_IDLE,      // Motors stopped
    STATE_RUNNING,   // Active episode
    STATE_RESETTING, // Slow return to position 0
    STATE_WAITING    // Waiting for pendulum to settle after reset
};

#endif // CONFIG_H
