#include <Arduino.h>
#include <Encoder.h>
#include "config.h"
#include "ai_inference.h"

Encoder myEnc(ENCODER_PIN_A, ENCODER_PIN_B);

// --- STATE ---
long  currentMotorSteps   = 0;
int   currentDirState     = LOW;
float targetSpeed         = 0.0f;

unsigned long lastStepMicros    = 0;
unsigned long stepIntervalMicros = 1000000;
unsigned long lastControlMicros = 0;
unsigned long waitingStartMillis = 0;
unsigned long lastResetStepMicros = 0;

long  encOffset           = 0;
float prev_enc_rads       = 0.0f;
float prev_motor_rads     = 0.0f;
float prev_filtered_action = 0.0f;
float filtered_enc_vel    = 0.0f;
float filtered_motor_vel  = 0.0f;

ControlMode currentMode = MODE_STANDALONE;
RobotState  robotState  = STATE_IDLE;

bool  pc_action_received = false;
float latest_pc_action   = 0.0f;

// --- FORWARD DECLARATIONS ---
void calibrateEncoder();
void handleSerialCommands();
void executeStepper(unsigned long currentMicros);
void executeReset(unsigned long currentMicros);
void executeWaiting();
void runControlLoop(unsigned long currentMicros);

// ============================================================
//  SETUP
// ============================================================
void setup() {
    Serial.begin(115200);

    pinMode(STEP_PIN, OUTPUT);
    pinMode(DIR_PIN,  OUTPUT);
    pinMode(EN_PIN,   OUTPUT);
    pinMode(LED_PIN,  OUTPUT);

    digitalWrite(EN_PIN,   LOW);
    digitalWrite(STEP_PIN, LOW);
    digitalWrite(DIR_PIN,  LOW);
    digitalWrite(LED_PIN,  LOW);

    init_ram_weights();
    calibrateEncoder();
    lastControlMicros = micros();
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
    unsigned long now = micros();
    handleSerialCommands();
    executeStepper(now);
    executeReset(now);
    executeWaiting();
    runControlLoop(now);
}

// ============================================================
//  CALIBRATION
// ============================================================
void calibrateEncoder() {
    Serial.println("\n[*] Leave pendulum hanging still - calibrating...");
    delay(2000);

    long sum = 0;
    for (int i = 0; i < 50; i++) {
        sum += myEnc.read();
        delay(10);
    }
    encOffset = sum / 50;

    Serial.print("[OK] Calibration done. Offset=");
    Serial.println(encOffset);
    Serial.println("STATE:IDLE");
}

// ============================================================
//  SERIAL COMMAND HANDLER
//  Commands:
//    S           - Start episode
//    P           - Pause / stop
//    R           - Reset motor to position 0
//    M:STANDALONE - Switch to edge-AI mode
//    M:PC_CONTROL - Switch to PC inference mode
//    ACT:<float>  - PC sends action (PC_CONTROL mode)
//    W:<layer>:<i>:<j>:<val> - Overwrite one weight in RAM
// ============================================================
void handleSerialCommands() {
    if (!Serial.available()) return;

    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    if (line == "S" || line == "s") {
        robotState = STATE_RUNNING;
        digitalWrite(LED_PIN, HIGH);
        prev_filtered_action = 0.0f;
        filtered_enc_vel     = 0.0f;
        filtered_motor_vel   = 0.0f;
        long ticks = myEnc.read();
        prev_enc_rads   = -(ticks - encOffset) * ENC_TICKS_TO_RADS;
        prev_motor_rads = currentMotorSteps * STEPS_TO_RADS;

    } else if (line == "P" || line == "p") {
        robotState = STATE_IDLE;
        targetSpeed = 0.0f;
        digitalWrite(LED_PIN, LOW);

    } else if (line == "R" || line == "r") {
        robotState  = STATE_RESETTING;
        targetSpeed = 0.0f;

    } else if (line == "M:STANDALONE") {
        currentMode = MODE_STANDALONE;

    } else if (line == "M:PC_CONTROL") {
        currentMode = MODE_PC_CONTROL;

    } else if (line.startsWith("ACT:")) {
        latest_pc_action  = line.substring(4).toFloat();
        pc_action_received = true;

    } else if (line.startsWith("W:")) {
        // W:<layer>:<i>:<j>:<value>
        // layer encoding: 0=obs_mean, 1=obs_var, 2=W1, 3=b1,
        //                 4=W2, 5=b2, 6=W3, 7=b3
        int layer = -1, i = -1, j = -1;
        float val = 0.0f;
        if (sscanf(line.c_str(), "W:%d:%d:%d:%f", &layer, &i, &j, &val) == 4) {
            switch (layer) {
                case 0: obs_mean_RAM[i]  = val; break;
                case 1: obs_var_RAM[i]   = val; break;
                case 2: W1_RAM[i][j]     = val; break;
                case 3: b1_RAM[i]        = val; break;
                case 4: W2_RAM[i][j]     = val; break;
                case 5: b2_RAM[i]        = val; break;
                case 6: W3_RAM[i][j]     = val; break;
                case 7: b3_RAM[i]        = val; break;
            }
        }
    }
}

// ============================================================
//  STEPPER EXECUTION (interrupt-free bit-banging)
// ============================================================
void executeStepper(unsigned long now) {
    if (robotState != STATE_RUNNING || targetSpeed == 0.0f) return;
    if (now - lastStepMicros < stepIntervalMicros) return;

    lastStepMicros = now;
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(2);
    digitalWrite(STEP_PIN, LOW);

    if (currentDirState == HIGH) currentMotorSteps++;
    else                         currentMotorSteps--;
}

// ============================================================
//  SLOW RESET TO POSITION 0
// ============================================================
void executeReset(unsigned long now) {
    if (robotState != STATE_RESETTING) return;

    if (currentMotorSteps == 0) {
        robotState = STATE_WAITING;
        waitingStartMillis = millis();
        Serial.println("STATE:WAITING");
        return;
    }

    if (now - lastResetStepMicros < 2500) return;
    lastResetStepMicros = now;

    int dir = (currentMotorSteps > 0) ? LOW : HIGH;
    digitalWrite(DIR_PIN, dir);
    delayMicroseconds(2);
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(2);
    digitalWrite(STEP_PIN, LOW);

    if (dir == LOW) currentMotorSteps--;
    else            currentMotorSteps++;
}

// ============================================================
//  WAITING FOR PENDULUM TO SETTLE
// ============================================================
void executeWaiting() {
    if (robotState != STATE_WAITING) return;
    if (millis() - waitingStartMillis < 4000) return;

    robotState = STATE_RUNNING;
    prev_filtered_action = 0.0f;
    Serial.println("STATE:READY");
}

// ============================================================
//  50 Hz CONTROL LOOP
// ============================================================
void runControlLoop(unsigned long now) {
    if (now - lastControlMicros < CONTROL_DT_MICROS) return;

    float dt = (now - lastControlMicros) / 1e6f;
    lastControlMicros = now;

    // --- Sensor reading ---
    long  ticks      = myEnc.read();
    float enc_rads   = -(ticks - encOffset) * ENC_TICKS_TO_RADS;
    float motor_rads = currentMotorSteps * STEPS_TO_RADS;

    // --- Velocity estimation (EMA filtered) ---
    float raw_enc_vel   = (enc_rads   - prev_enc_rads)   / dt;
    float raw_motor_vel = (motor_rads - prev_motor_rads) / dt;

    filtered_enc_vel   = (1.0f - FILTER_ENC) * filtered_enc_vel   + FILTER_ENC * raw_enc_vel;
    filtered_motor_vel = (1.0f - FILTER_MOT) * filtered_motor_vel + FILTER_MOT * raw_motor_vel;

    prev_enc_rads   = enc_rads;
    prev_motor_rads = motor_rads;

    // --- Telemetry (STANDALONE mode only) ---
    if (currentMode == MODE_STANDALONE) {
        Serial.print("TEL:");
        Serial.print(currentMotorSteps);
        Serial.print(",");
        Serial.println(ticks);
    }

    if (robotState != STATE_RUNNING) return;

    // --- Build observation vector ---
    float obs[6] = {
        motor_rads,
        filtered_motor_vel,
        cosf(enc_rads),
        sinf(enc_rads),
        filtered_enc_vel,
        prev_filtered_action
    };

    // --- Compute action ---
    float raw_action = 0.0f;

    if (currentMode == MODE_STANDALONE) {
        raw_action = compute_ai_action(obs);

    } else {  // MODE_PC_CONTROL
        // Send observation to PC, wait for action reply (timeout 15 ms)
        Serial.print("OBS:");
        Serial.print(obs[0], 4); Serial.print(",");
        Serial.print(obs[1], 4); Serial.print(",");
        Serial.print(obs[2], 4); Serial.print(",");
        Serial.print(obs[3], 4); Serial.print(",");
        Serial.print(obs[4], 4); Serial.print(",");
        Serial.println(obs[5], 4);

        pc_action_received = false;
        unsigned long waitStart = micros();
        while (!pc_action_received && (micros() - waitStart < 15000)) {
            handleSerialCommands();
        }
        raw_action = pc_action_received ? latest_pc_action : 0.0f;
    }

    // --- Action filter ---
    float filtered_action = FILTER_ALPHA * prev_filtered_action +
                            (1.0f - FILTER_ALPHA) * raw_action;
    prev_filtered_action = filtered_action;
    targetSpeed = filtered_action * MAX_VELOCITY;

    // --- Drive stepper ---
    if (targetSpeed != 0.0f) {
        stepIntervalMicros = (unsigned long)(1e6f / fabsf(targetSpeed));
        int newDir = (targetSpeed > 0.0f) ? HIGH : LOW;
        if (newDir != currentDirState) {
            digitalWrite(DIR_PIN, newDir);
            currentDirState = newDir;
            delayMicroseconds(2);
        }
    }
}
