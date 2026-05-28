"""
PhysicalFurutaEnv  –  Real-Robot Gymnasium Environment
=======================================================
Wraps the physical Furuta pendulum (Teensy 4.x + stepper motor + encoder)
as a Gymnasium environment identical in interface to SimMotorRLEnv.

Serial protocol (see firmware/main.cpp):
  PC → Teensy:  S / P / R / M:PC_CONTROL / ACT:<f> / W:<layer>:<i>:<j>:<val>
  Teensy → PC:  OBS:<f>,<f>,<f>,<f>,<f>,<f>   (observation)
                STATE:WAITING / STATE:READY     (reset handshake)
                TEL:<steps>,<ticks>             (standalone telemetry)
"""
import math
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import serial


class PhysicalFurutaEnv(gym.Env):

    def __init__(self, serial_port: str, baud_rate: int = 115200):
        super().__init__()

        self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.max_motor_steps   = 4800.0
        self.steps_to_rads     = (2 * math.pi) / 3200.0
        self.max_motor_rad     = self.max_motor_steps * self.steps_to_rads

        self.prev_raw_action   = 0.0
        self.steps_taken       = 0
        self.max_episode_steps = 800

        self.current_episode_reward = 0.0

        # Set by train_on_robot.py so weights can be streamed after each reset
        self.model = None

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.steps_taken > 0:
            print(
                f"\n{'='*50}\n"
                f"[★] Episode done  (steps={self.steps_taken}) | "
                f"reward={self.current_episode_reward:.2f}\n"
                f"{'='*50}"
            )

        print("\n[*] Sending RESET to Teensy...")
        self.ser.reset_input_buffer()

        # Switch to PC_CONTROL and start motor reset
        self.ser.write(b"M:PC_CONTROL\n")
        time.sleep(0.05)
        self.ser.write(b"R\n")

        waiting_attempts = 0

        # --- Phase 1: wait for motor to reach position 0 ---
        while True:
            line = self._readline()
            if line and not line.startswith(("OBS:", "TEL:")):
                print(f"[TEENSY] {line}")
            if "STATE:WAITING" in line:
                print("[OK] Motor at 0. Pendulum settling...")
                break
            # Re-send commands if Teensy did not acknowledge
            if line.startswith("TEL:") or not line:
                waiting_attempts += 1
                if waiting_attempts % 10 == 0:
                    self.ser.write(b"M:PC_CONTROL\n")
                    self.ser.write(b"R\n")
            time.sleep(0.01)

        # --- Stream updated weights while pendulum is settling ---
        if self.model is not None:
            self._send_weights_to_teensy()

        print("[*] Waiting for STATE:READY...")

        # --- Phase 2: wait for pendulum to settle ---
        while True:
            line = self._readline()
            if line and not line.startswith(("OBS:", "TEL:")):
                print(f"[TEENSY] {line}")
            if "STATE:READY" in line:
                print("[OK] Pendulum settled. Starting episode.")
                break
            time.sleep(0.01)

        # Reset episode state
        self.prev_raw_action        = 0.0
        self.steps_taken            = 0
        self.current_episode_reward = 0.0

        self.ser.reset_input_buffer()
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action):
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        self.ser.write(f"ACT:{raw_action:.4f}\n".encode())

        obs = self._get_obs()
        motor_pos, motor_vel, enc_cos, _, enc_vel, _ = obs

        # Reward function (identical to SimMotorRLEnv)
        cost = (
            1.000 * (1.0 + enc_cos) +
            0.150 * abs(motor_pos) +
            0.020 * enc_vel**2 +
            0.005 * abs(motor_vel) +
            0.005 * (raw_action - self.prev_raw_action)**2
        )
        reward = math.exp(-cost)

        self.current_episode_reward += reward
        self.prev_raw_action         = raw_action
        self.steps_taken            += 1

        terminated = False
        if abs(motor_pos) > self.max_motor_rad:
            print(f"[!] Bounds violation – motor_pos={motor_pos:.2f} rad")
            reward                      -= 10.0
            self.current_episode_reward -= 10.0
            terminated                   = True

        truncated = bool(self.steps_taken >= self.max_episode_steps)
        return obs, float(reward), terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _readline(self) -> str:
        try:
            return self.ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _get_obs(self) -> np.ndarray:
        """Block until Teensy sends a valid OBS: line."""
        while True:
            line = self._readline()
            if not line.startswith("OBS:"):
                continue
            parts = line[4:].split(",")
            if len(parts) != 6:
                continue
            try:
                return np.array([float(x) for x in parts], dtype=np.float32)
            except ValueError:
                continue

    def _send_weights_to_teensy(self):
        """Stream the current policy weights into the Teensy's RAM buffers."""
        print("[*] Streaming weights to Teensy RAM...")
        try:
            params = self.model.policy.state_dict()

            # Observation normalisation statistics
            if hasattr(self, "obs_rms"):
                mean = self.obs_rms.mean
                var  = self.obs_rms.var
                for i in range(6):
                    self.ser.write(f"W:0:{i}:0:{mean[i]:.6f}\n".encode())
                    self.ser.write(f"W:1:{i}:0:{var[i]:.6f}\n".encode())

            # Layer 1
            w1 = params["mlp_extractor.policy_net.0.weight"].cpu().numpy()
            b1 = params["mlp_extractor.policy_net.0.bias"].cpu().numpy()
            for i in range(64):
                self.ser.write(f"W:3:{i}:0:{b1[i]:.6f}\n".encode())
                for j in range(6):
                    self.ser.write(f"W:2:{i}:{j}:{w1[i][j]:.6f}\n".encode())
                time.sleep(0.001)

            # Layer 2
            w2 = params["mlp_extractor.policy_net.2.weight"].cpu().numpy()
            b2 = params["mlp_extractor.policy_net.2.bias"].cpu().numpy()
            for i in range(64):
                self.ser.write(f"W:5:{i}:0:{b2[i]:.6f}\n".encode())
                for j in range(64):
                    self.ser.write(f"W:4:{i}:{j}:{w2[i][j]:.6f}\n".encode())
                if i % 10 == 0:
                    time.sleep(0.002)

            # Output layer
            w3 = params["action_net.weight"].cpu().numpy()
            b3 = params["action_net.bias"].cpu().numpy()
            self.ser.write(f"W:7:0:0:{b3[0]:.6f}\n".encode())
            for j in range(64):
                self.ser.write(f"W:6:0:{j}:{w3[0][j]:.6f}\n".encode())

            print("[OK] Weight transfer complete.")
        except Exception as exc:
            print(f"[ERROR] Weight transfer failed: {exc}")

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #
    def close(self):
        self.ser.write(b"P\n")
        self.ser.close()
