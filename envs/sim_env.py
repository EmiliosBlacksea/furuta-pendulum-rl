"""
SimMotorRLEnv  –  Furuta Pendulum Simulation Environment
=========================================================
Simulates a stepper-motor-driven Furuta (rotary) pendulum at 50 Hz.
Key sim-to-real features:
  - Encoder & motor position quantisation
  - EMA velocity filters matching the Teensy firmware
  - Domain randomisation (physics params, gravity tilt, latency)
  - Optional Gaussian sensor noise and process disturbances
"""
import math
import collections

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimMotorRLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self):
        super().__init__()

        # Base (nominal) physical parameters
        self.base_m   = 0.048      # pendulum bob mass [kg]
        self.base_r   = 0.12       # motor arm length [m]
        self.base_l   = 0.1669     # pendulum length [m]
        self.base_J   = 0.001336   # pendulum moment of inertia [kg·m²]
        self.base_b   = 0.001      # pendulum viscous damping [N·m·s]
        self.base_tau = 0.005109   # motor first-order time constant [s]
        self.g        = 9.81       # gravitational acceleration [m/s²]

        # Working copies (overwritten each reset by domain randomisation)
        self.m = self.base_m
        self.r = self.base_r
        self.l = self.base_l
        self.J = self.base_J
        self.b = self.base_b
        self.tau = self.base_tau

        # Control / timing  (must match firmware config.h)
        self.dt               = 1.0 / 50.0            # 50 Hz
        self.max_velocity     = 3000.0                 # steps/sec
        self.steps_to_rads    = (2 * math.pi) / 3200.0
        self.max_motor_steps  = 4800.0
        self.max_episode_steps = 800

        # Action filter  (must match FILTER_ALPHA in config.h)
        self.filter_alpha        = 0.3
        self.prev_filtered_action = 0.0
        self.prev_raw_action     = 0.0

        # Action latency buffer (0 or 1 steps, randomised when dr_scale > 0.5)
        self.latency_steps  = 0
        self.current_latency = 0
        self.action_buffer  = collections.deque(maxlen=5)

        # Quantisation (sim-to-real: encoder & stepper discrete steps)
        self.encoder_cpr  = 2400.0
        self.enc_step_size = (2 * math.pi) / self.encoder_cpr

        # Velocity EMA filters  (must match FILTER_ENC / FILTER_MOT in config.h)
        self.filter_enc = 0.80
        self.filter_mot = 0.80

        self.sim_filtered_enc_vel   = 0.0
        self.sim_filtered_motor_vel = 0.0
        self.prev_quantized_enc_pos   = 0.0
        self.prev_quantized_motor_pos = 0.0

        # Noise / domain-randomisation scales (set by curriculum callback)
        self.noise_motor_pos = 0.0
        self.noise_enc_pos   = 0.001
        self.noise_process   = 0.0
        self.dr_scale        = 0.0   # 0 = no DR, 1 = full DR

        # Sudden-death zone (enc_cos < threshold = near bottom = penalised)
        self.balance_threshold  = -0.60
        self.is_in_balance      = False
        self.spawned_in_balance = False

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        s = self.dr_scale

        # Domain randomisation of physical parameters
        self.m   = self.base_m   * self.np_random.uniform(1.0 - 0.20 * s, 1.0 + 0.20 * s)
        self.l   = self.base_l   * self.np_random.uniform(1.0 - 0.20 * s, 1.0 + 0.20 * s)
        self.J   = self.base_J   * self.np_random.uniform(1.0 - 0.35 * s, 1.0 + 0.35 * s)
        self.b   = self.base_b   * self.np_random.uniform(1.0 - 0.80 * s, 1.0 + 0.80 * s)
        self.r   = self.base_r   * self.np_random.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s)
        self.tau = self.base_tau * self.np_random.uniform(1.0 - 0.40 * s, 1.0 + 0.40 * s)

        # Random gravity tilt (simulates mounting imperfection)
        max_tilt = math.radians(4.0)
        self.gravity_tilt = self.np_random.uniform(-max_tilt, max_tilt) * s

        # Random communication latency
        self.current_latency = 1 if (s > 0.5 and self.np_random.random() < 0.5) else 0

        # Initial conditions
        self.motor_pos = self.np_random.uniform(-2.5 * math.pi, 2.5 * math.pi)
        self.motor_vel = 0.0
        self.enc_pos   = self.np_random.uniform(-math.pi, math.pi)
        self.enc_vel   = 0.0

        self.steps_taken          = 0
        self.prev_filtered_action = 0.0
        self.prev_raw_action      = 0.0

        # Reset sim-to-real filters
        self.sim_filtered_enc_vel   = 0.0
        self.sim_filtered_motor_vel = 0.0
        self.prev_quantized_motor_pos = (
            round(self.motor_pos / self.steps_to_rads) * self.steps_to_rads
        )
        self.prev_quantized_enc_pos = (
            round(self.enc_pos / self.enc_step_size) * self.enc_step_size
        )

        # Sudden-death grace period
        if math.cos(self.enc_pos) < self.balance_threshold:
            self.is_in_balance      = True
            self.spawned_in_balance = True
        else:
            self.is_in_balance      = False
            self.spawned_in_balance = False

        self.action_buffer.clear()
        self.action_buffer.extend([0.0] * 5)

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    #  Observation (with quantisation + EMA, matching Teensy exactly)     #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        # Encoder: add small noise, then quantise to CPR ticks
        noisy_enc_pos    = self.enc_pos + self.np_random.normal(0, self.noise_enc_pos)
        quantized_enc_pos = round(noisy_enc_pos / self.enc_step_size) * self.enc_step_size

        # Motor: add small noise, then quantise to stepper steps
        noisy_motor_pos    = self.motor_pos + self.np_random.normal(0, self.noise_motor_pos)
        quantized_motor_pos = round(noisy_motor_pos / self.steps_to_rads) * self.steps_to_rads

        # Velocity by finite difference on quantised positions (same as firmware)
        raw_enc_vel   = (quantized_enc_pos   - self.prev_quantized_enc_pos)   / self.dt
        raw_motor_vel = (quantized_motor_pos - self.prev_quantized_motor_pos) / self.dt

        # EMA filter (same constants as FILTER_ENC / FILTER_MOT in config.h)
        self.sim_filtered_enc_vel = (
            (1.0 - self.filter_enc) * self.sim_filtered_enc_vel +
            self.filter_enc * raw_enc_vel
        )
        self.sim_filtered_motor_vel = (
            (1.0 - self.filter_mot) * self.sim_filtered_motor_vel +
            self.filter_mot * raw_motor_vel
        )

        self.prev_quantized_enc_pos   = quantized_enc_pos
        self.prev_quantized_motor_pos = quantized_motor_pos

        return np.array([
            quantized_motor_pos,
            self.sim_filtered_motor_vel,
            math.cos(quantized_enc_pos),
            math.sin(quantized_enc_pos),
            self.sim_filtered_enc_vel,
            self.prev_filtered_action,
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action):
        raw_action = float(np.clip(action[0], -1.0, 1.0))

        # Action filter + latency buffer
        filtered_action = (
            self.filter_alpha * self.prev_filtered_action +
            (1.0 - self.filter_alpha) * raw_action
        )
        self.action_buffer.append(filtered_action)
        delayed_action = self.action_buffer[-1 - self.current_latency]

        # --- Motor dynamics (first-order lag) ---
        target_vel_rads = delayed_action * self.max_velocity * self.steps_to_rads
        old_motor_vel   = self.motor_vel
        self.motor_vel  = target_vel_rads + (old_motor_vel - target_vel_rads) * math.exp(-self.dt / self.tau)
        motor_accel     = (self.motor_vel - old_motor_vel) / self.dt
        self.motor_pos += self.motor_vel * self.dt

        # --- Pendulum dynamics (full Furuta equations of motion) ---
        # Per-step dynamic friction randomisation (sim-to-real)
        dynamic_b = self.b * self.np_random.uniform(0.2, 2.0)

        gravity_torque     = -self.m * self.g * self.l * math.sin(self.enc_pos + self.gravity_tilt)
        friction_torque    = -dynamic_b * self.enc_vel
        inertial_torque    =  self.m * self.l * self.r * motor_accel * math.cos(self.enc_pos)
        centrifugal_torque =  self.m * self.l**2 * self.motor_vel**2 * math.sin(self.enc_pos) * math.cos(self.enc_pos)
        process_noise      =  self.np_random.normal(0, self.noise_process)

        denom      = self.J + self.m * self.l**2
        enc_acc    = (gravity_torque + friction_torque + inertial_torque + centrifugal_torque + process_noise) / denom
        self.enc_vel += enc_acc * self.dt
        self.enc_pos += self.enc_vel * self.dt

        # Random kick perturbation (1.5% chance per step – robustness training)
        if self.np_random.random() < 0.005:
            self.enc_vel += self.np_random.uniform(-1.0, 1.0)

        self.steps_taken += 1
        obs     = self._get_obs()
        enc_cos = obs[2]

        # --- Reward (exponential cost shaping) ---
        cost = (
            1.000 * (1.0 + enc_cos) +                              # upright pose
            0.150 * abs(self.motor_pos) +                          # stay near centre
            0.020 * self.enc_vel**2 +                              # low pendulum speed
            0.005 * self.motor_vel +                               # smooth motor motion
            0.005 * (raw_action - self.prev_raw_action)**2         # low action jerk
        )
        reward = math.exp(-cost)

        self.prev_filtered_action = filtered_action
        self.prev_raw_action      = raw_action

        # --- Termination ---
        max_motor_rad = self.max_motor_steps * self.steps_to_rads
        terminated = False
        if abs(self.motor_pos) > max_motor_rad:
            reward    -= 1.0
            terminated = True

        truncated = bool(self.steps_taken >= self.max_episode_steps)

        return obs, float(reward), terminated, truncated, {}
