#ifndef AI_INFERENCE_H
#define AI_INFERENCE_H

// Copy factory weights from flash (model_weights.h) into RAM so they can be
// overwritten at runtime via Serial during PC-control / fine-tuning.
void init_ram_weights();

// Run one forward pass of the MLP policy. obs[6] is consumed in-place
// (normalized), returns a clipped action in [-1.0, 1.0].
float compute_ai_action(float obs[6]);

// RAM copies exposed so main.cpp can overwrite them via the W: serial protocol.
extern float obs_mean_RAM[6];
extern float obs_var_RAM[6];
extern float W1_RAM[64][6];
extern float b1_RAM[64];
extern float W2_RAM[64][64];
extern float b2_RAM[64];
extern float W3_RAM[1][64];
extern float b3_RAM[1];

#endif // AI_INFERENCE_H
