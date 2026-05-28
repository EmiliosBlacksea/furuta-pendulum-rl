#include "ai_inference.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

// RAM buffers (writable at runtime via Serial W: protocol)
float obs_mean_RAM[6];
float obs_var_RAM[6];
float W1_RAM[64][6];
float b1_RAM[64];
float W2_RAM[64][64];
float b2_RAM[64];
float W3_RAM[1][64];
float b3_RAM[1];

void init_ram_weights() {
    memcpy(obs_mean_RAM, obs_mean, sizeof(obs_mean));
    memcpy(obs_var_RAM,  obs_var,  sizeof(obs_var));
    memcpy(W1_RAM, W1, sizeof(W1));
    memcpy(b1_RAM, b1, sizeof(b1));
    memcpy(W2_RAM, W2, sizeof(W2));
    memcpy(b2_RAM, b2, sizeof(b2));
    memcpy(W3_RAM, W3, sizeof(W3));
    memcpy(b3_RAM, b3, sizeof(b3));
}

float compute_ai_action(float obs[6]) {
    float h1[64];
    float h2[64];

    // 1. Normalize observations
    for (int i = 0; i < 6; i++) {
        obs[i] = (obs[i] - obs_mean_RAM[i]) / sqrtf(obs_var_RAM[i] + 1e-8f);
        obs[i] = constrain(obs[i], -10.0f, 10.0f);
    }

    // 2. Layer 1: linear + tanh
    for (int i = 0; i < 64; i++) {
        float sum = b1_RAM[i];
        for (int j = 0; j < 6; j++) sum += obs[j] * W1_RAM[i][j];
        h1[i] = tanhf(sum);
    }

    // 3. Layer 2: linear + tanh
    for (int i = 0; i < 64; i++) {
        float sum = b2_RAM[i];
        for (int j = 0; j < 64; j++) sum += h1[j] * W2_RAM[i][j];
        h2[i] = tanhf(sum);
    }

    // 4. Output layer (linear)
    float raw_action = b3_RAM[0];
    for (int j = 0; j < 64; j++) raw_action += h2[j] * W3_RAM[0][j];

    // 5. Clip to valid action range
    return constrain(raw_action, -1.0f, 1.0f);
}
