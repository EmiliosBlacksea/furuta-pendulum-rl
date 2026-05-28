"""
visualize.py  –  Visualise a trained policy in the simulator
============================================================
Loads a saved PPO model + VecNormalize stats and renders the Furuta
pendulum in a Pygame window at 50 FPS.

Usage:
    python scripts/visualize.py
"""
import math
import os

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import SimMotorRLEnv

# ------------------------------------------------------------------ #
#  Configuration – edit these to point at your saved model           #
# ------------------------------------------------------------------ #
ALGO_NAME   = "PPO_dynfric_dyngrav_3000_alpha03_08"
MODELS_DIR  = "./saved_models/"
MODEL_PATH  = os.path.join(MODELS_DIR, ALGO_NAME, "best_model.zip")
STATS_PATH  = os.path.join(MODELS_DIR, ALGO_NAME, "best_vec_normalize.pkl")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    print(f"[*] Loading {ALGO_NAME}...")

    env = DummyVecEnv([SimMotorRLEnv])
    env = VecNormalize.load(STATS_PATH, env)
    env.training   = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env)

    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Furuta Pendulum – {ALGO_NAME}")
    clock  = pygame.font.SysFont("Arial", 18)
    font   = pygame.font.SysFont("Arial", 18)

    WHITE     = (255, 255, 255)
    GRAY      = (200, 200, 200)
    DARK_GRAY = (50,  50,  50)
    BLUE      = (50, 100, 255)
    RED       = (255, 50,  50)
    BLACK     = (0,   0,   0)

    CX, CY         = 400, 300   # pivot (motor axis) in screen coords
    MOTOR_PX       = 120        # motor arm length [px]
    PENDULUM_PX    = 160        # pendulum length [px]

    obs = env.reset()
    running        = True
    episode_reward = 0.0
    step_count     = 0
    fps_clock      = pygame.time.Clock()

    print("[OK] Window open. Press ESC or close to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, _ = env.step(action)
        episode_reward += reward[0]
        step_count     += 1

        motor_pos = env.get_attr("motor_pos")[0]
        enc_pos   = env.get_attr("enc_pos")[0]

        # Motor arm endpoint (pivot of pendulum)
        pivot_x = CX + MOTOR_PX    * math.sin(motor_pos)
        pivot_y = CY - MOTOR_PX    * math.cos(motor_pos)

        # Pendulum bob position
        pend_x  = pivot_x + PENDULUM_PX * math.sin(enc_pos)
        pend_y  = pivot_y + PENDULUM_PX * math.cos(enc_pos)

        screen.fill(WHITE)
        pygame.draw.circle(screen, GRAY,      (CX, CY),              MOTOR_PX, 1)
        pygame.draw.line(  screen, BLUE,      (CX, CY),              (int(pivot_x), int(pivot_y)), 6)
        pygame.draw.circle(screen, DARK_GRAY, (CX, CY),              10)
        pygame.draw.line(  screen, RED,       (int(pivot_x), int(pivot_y)), (int(pend_x),  int(pend_y)),  4)
        pygame.draw.circle(screen, BLACK,     (int(pivot_x), int(pivot_y)), 6)
        pygame.draw.circle(screen, RED,       (int(pend_x),  int(pend_y)),  12)

        motor_deg = math.degrees(motor_pos)
        enc_deg   = math.degrees(enc_pos) % 360

        screen.blit(font.render(f"Model: {ALGO_NAME}",                       True, BLACK), (20, 20))
        screen.blit(font.render(f"Motor:    {motor_deg:>6.1f}°",             True, BLUE),  (20, 45))
        screen.blit(font.render(f"Pendulum: {enc_deg:>6.1f}°",              True, RED),   (20, 70))
        screen.blit(font.render(f"Step {step_count}  |  Reward {episode_reward:.1f}", True, BLACK), (20, 95))

        pygame.display.flip()

        if dones[0]:
            print(f"Episode done. Reward: {episode_reward:.2f}")
            obs            = env.reset()
            episode_reward = 0.0
            step_count     = 0

        fps_clock.tick(50)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
