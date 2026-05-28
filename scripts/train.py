"""
train.py  –  Simulation Training with Curriculum Noise
=======================================================
Trains a PPO policy in SimMotorRLEnv using:
  • Gradual sensor-noise curriculum (no noise → full noise over 1M steps)
  • Domain randomisation (physics params, gravity tilt, action latency)
  • EvalCallback that saves every checkpoint and the global best model
  • TensorBoard logging

Usage:
    python scripts/train.py
"""
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from envs import SimMotorRLEnv

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
ALGO_NAME        = "PPO_dynfric_dyngrav_3000_alpha03_08"
TOTAL_TIMESTEPS  = 2_200_000
START_NOISE_STEP = 10_000    # curriculum noise starts here
END_NOISE_STEP   = 1_000_000 # curriculum noise reaches maximum here
EVAL_FREQ        = 10_000

MODELS_DIR       = "./saved_models/"
TENSORBOARD_DIR  = "./tensorboard_logs/"

TARGET_NOISE = {
    "motor_pos": 0.01,
    "enc_pos":   0.01,
    "process":   0.30,
}

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps":       2048,
    "batch_size":    256,
    "gamma":         0.99,
    "ent_coef":      0.005,
}


def make_env():
    return Monitor(SimMotorRLEnv())


# ------------------------------------------------------------------ #
#  Curriculum callback: ramps sensor noise from 0 → max              #
# ------------------------------------------------------------------ #
class NoiseCurriculumCallback(BaseCallback):
    def __init__(self, target_noise, start_step, end_step, verbose=0):
        super().__init__(verbose)
        self.target_noise = target_noise
        self.start_step   = start_step
        self.end_step     = end_step

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t <= self.start_step:
            frac = 0.0
        elif t >= self.end_step:
            frac = 1.0
        else:
            frac = (t - self.start_step) / (self.end_step - self.start_step)

        self.training_env.set_attr("noise_motor_pos", frac * self.target_noise["motor_pos"])
        self.training_env.set_attr("noise_enc_pos",   frac * self.target_noise["enc_pos"])
        self.training_env.set_attr("noise_process",   frac * self.target_noise["process"])
        self.training_env.set_attr("dr_scale",        frac)

        self.logger.record("curriculum/difficulty_fraction",   frac)
        self.logger.record("curriculum/encoder_noise_rads", frac * self.target_noise["enc_pos"])

        if t % 100_000 == 0:
            print(f"[Step {t}] Curriculum difficulty: {frac*100:.1f}%")

        return True


# ------------------------------------------------------------------ #
#  Eval callback: saves every checkpoint + global best               #
# ------------------------------------------------------------------ #
class CheckpointedEvalCallback(EvalCallback):
    def __init__(self, train_env, start_saving_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_env         = train_env
        self.start_saving_step = start_saving_step
        self.all_evals_dir     = os.path.join(self.best_model_save_path, "all_evals")
        os.makedirs(self.all_evals_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_saving_step:
            return True

        # Sync normalisation statistics to eval env before evaluating
        if hasattr(self.train_env, "obs_rms") and hasattr(self.eval_env, "obs_rms"):
            self.eval_env.obs_rms = self.train_env.obs_rms

        old_best = self.best_mean_reward
        result   = super()._on_step()

        # Save a checkpoint at every evaluation
        if self.n_calls % self.eval_freq == 0:
            tag = f"step_{self.num_timesteps}"
            self.model.save(os.path.join(self.all_evals_dir, f"model_{tag}"))
            self.train_env.save(os.path.join(self.all_evals_dir, f"vec_normalize_{tag}.pkl"))

        # Save VecNormalize alongside the best model whenever it improves
        if self.best_mean_reward > old_best and self.best_model_save_path is not None:
            self.train_env.save(
                os.path.join(self.best_model_save_path, "best_vec_normalize.pkl")
            )

        return result


# ------------------------------------------------------------------ #
#  Main training routine                                              #
# ------------------------------------------------------------------ #
def main():
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    save_path = os.path.join(MODELS_DIR, ALGO_NAME)
    os.makedirs(save_path, exist_ok=True)

    # Training environment
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Evaluation environment (always at full difficulty)
    eval_env = DummyVecEnv([make_env])
    eval_env.set_attr("noise_motor_pos", TARGET_NOISE["motor_pos"])
    eval_env.set_attr("noise_enc_pos",   TARGET_NOISE["enc_pos"])
    eval_env.set_attr("noise_process",   TARGET_NOISE["process"])
    eval_env.set_attr("dr_scale",        1.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    noise_cb = NoiseCurriculumCallback(
        target_noise=TARGET_NOISE,
        start_step=START_NOISE_STEP,
        end_step=END_NOISE_STEP,
    )
    eval_cb = CheckpointedEvalCallback(
        train_env=train_env,
        start_saving_step=START_NOISE_STEP,
        eval_env=eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy", train_env,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        **PPO_KWARGS,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([noise_cb, eval_cb]),
        tb_log_name=ALGO_NAME,
    )

    model.save(os.path.join(save_path, "final_model"))
    train_env.save(os.path.join(save_path, "final_vec_normalize.pkl"))
    print(f"[OK] Training complete – model saved to {save_path}/")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
