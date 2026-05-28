"""
finetune_robot.py  –  Real-Robot Linear Probing / Fine-Tuning
==============================================================
Loads a pre-trained PPO model, freezes the shared feature-extractor
(layers 1 & 2), and fine-tunes only the output heads (action_net /
value_net) on the physical robot via PC_CONTROL mode.

Weights are streamed to the Teensy after every episode reset so the
robot can eventually run standalone with the updated policy.

Usage:
    python scripts/finetune_robot.py
"""
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs import PhysicalFurutaEnv

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
SERIAL_PORT = "COM3"   # ← change to your port (e.g. /dev/ttyUSB0)

BASE_MODEL  = "PPO_dynfric_dyngrav_3000_alpha03_08"
MODELS_DIR  = "./saved_models/"

PRETRAINED_MODEL_PATH   = os.path.join(MODELS_DIR, BASE_MODEL, "best_model.zip")
PRETRAINED_VECNORM_PATH = os.path.join(MODELS_DIR, BASE_MODEL, "best_vec_normalize.pkl")

FINETUNE_TIMESTEPS = 200_000
CHECKPOINT_FREQ    = 1_000  # steps between auto-saves


def main():
    print("=== REAL-WORLD FINE-TUNING ===")

    os.makedirs(os.path.join(MODELS_DIR, BASE_MODEL), exist_ok=True)
    os.makedirs("./tensorboard_real_logs/", exist_ok=True)
    os.makedirs("./saved_models/real_checkpoints/lastlayer/", exist_ok=True)

    # --- Environment setup ---
    raw_env = PhysicalFurutaEnv(SERIAL_PORT)
    time.sleep(4)  # Wait for USB serial to stabilise

    raw_env.ser.write(b"M:PC_CONTROL\n")
    raw_env.ser.write(b"S\n")
    print("[*] Teensy set to PC_CONTROL mode.")

    env = DummyVecEnv([lambda: Monitor(raw_env)])

    if os.path.exists(PRETRAINED_VECNORM_PATH):
        print(f"[*] Loading VecNormalize stats from {PRETRAINED_VECNORM_PATH}")
        env = VecNormalize.load(PRETRAINED_VECNORM_PATH, env)
        env.training    = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Share obs_rms with the physical env so it can stream normalisation stats
    raw_env.obs_rms = env.obs_rms

    # --- Model setup ---
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Pre-trained model not found: {PRETRAINED_MODEL_PATH}")

    print(f"[*] Loading model: {PRETRAINED_MODEL_PATH}")
    model = PPO.load(
        PRETRAINED_MODEL_PATH, env=env,
        learning_rate=1e-4,
        tensorboard_log="./tensorboard_real_logs/",
    )

    # Freeze shared layers; fine-tune only the output heads
    print("[*] Freezing feature extractor (layers 1 & 2)...")
    for name, param in model.policy.named_parameters():
        if "action_net" in name or "value_net" in name:
            param.requires_grad = True
            print(f"  [TRAIN] {name}")
        else:
            param.requires_grad = False
    print("[OK] Ready for linear fine-tuning.")

    # Give the physical env a reference to the model so it can stream weights
    raw_env.model = model

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./saved_models/real_checkpoints/lastlayer/",
        name_prefix="ppo_real_finetune",
        save_vecnormalize=True,
    )

    # --- Fine-tuning loop ---
    try:
        model.learn(
            total_timesteps=FINETUNE_TIMESTEPS,
            log_interval=1,
            tb_log_name="PPO_Real_FineTune",
            callback=checkpoint_cb,
        )
        model.save("./saved_models/ppo_furuta_real_world_final")
        env.save("./saved_models/vec_normalize_real_world_final.pkl")
        print("[OK] Fine-tuning complete. Final model saved.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Saving checkpoint...")
        model.save("./saved_models/ppo_furuta_real_world_interrupted")
        env.save("./saved_models/vec_normalize_real_world_interrupted.pkl")
        print("[OK] Checkpoint saved.")

    # --- Post-training mode selection ---
    print("\nWhat next?")
    print("  1 – Switch Teensy to STANDALONE mode (runs new weights autonomously)")
    print("  2 – Pause and exit")
    choice = input("Choice [1/2]: ").strip()

    if choice == "1":
        raw_env.ser.write(b"M:STANDALONE\n")
        raw_env.ser.write(b"S\n")
        print("[OK] Teensy running autonomously with fine-tuned weights.")
    else:
        raw_env.ser.write(b"P\n")
        print("[*] System paused.")

    raw_env.close()


if __name__ == "__main__":
    main()
