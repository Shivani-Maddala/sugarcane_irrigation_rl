"""
Sugarcane irrigation environment (Gymnasium interface).

State s_t = [P_t (7-day rainfall forecast), h_t, h_min, h_max, H_p, C_t]
  - P_t: forecast rainfall for the next 7 days, mm (from WeatherDataProvider)
  - h_t: current available soil water in the root zone, mm
  - h_min, h_max, H_p: stage-dependent water depth bands (config.py)
  - C_t: crop-condition level from the CNN (0=healthy, 1=moderate, 2=severe)

Action a_t in {0, 1, 2}: supply 0%, 50%, or 100% of the irrigation quota
(quota = amount needed to bring h_t up to h_max), same 3-action design as
the reference paper.

Reward r_t = r0 * r1 * r2 * r3 (see config.py for what each term represents).

This environment reuses the reference paper's water-balance and crop-water-
production-function equations (their Eqs. 4-18), swapping in sugarcane's
FAO Kc/Ky values and growth stages instead of rice's.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import config


class SugarcaneIrrigationEnv(gym.Env):
    def __init__(self, weather_provider, crop_condition_provider=None):
        super().__init__()
        self.weather = weather_provider
        # crop_condition_provider: callable(t) -> int in {0,1,2}. If None,
        # defaults to "always healthy" (0) -- lets the RL side be smoke-tested
        # independently of the CNN before the two are wired together in Step 7.
        self.crop_condition_provider = crop_condition_provider or (lambda t: 0)

        self.action_space = spaces.Discrete(config.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.STATE_DIM,), dtype=np.float32
        )

        self._build_stage_schedule()
        self.reset()

    def _build_stage_schedule(self):
        """Expands config.GROWTH_STAGES into a per-day list of (kc, ky, stage_name)."""
        schedule = []
        stages = config.GROWTH_STAGES
        # Interpolate Kc through the "development" stage between initial and mid Kc.
        kc_initial = stages[0]["kc"]
        kc_mid = stages[2]["kc"]
        for stage in stages:
            if stage["kc"] is None:  # development stage
                for d in range(stage["length_days"]):
                    frac = d / max(stage["length_days"] - 1, 1)
                    kc = kc_initial + frac * (kc_mid - kc_initial)
                    schedule.append((kc, stage["ky"], stage["name"]))
            else:
                for _ in range(stage["length_days"]):
                    schedule.append((stage["kc"], stage["ky"], stage["name"]))
        self.stage_schedule = schedule  # length == config.SEASON_LENGTH_DAYS

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.h = config.H_MAX_MM * 0.7  # start at 70% of field capacity, a reasonable non-extreme initial condition
        self.cumulative_irrigation_mm = 0.0
        self.cumulative_drainage_mm = 0.0
        self.cumulative_actual_et = 0.0
        self.cumulative_max_et = 0.0
        return self._get_state(), {}

    def _reference_et0(self, t):
        """Simplified ET0 proxy. A full FAO56 Penman-Monteith calculation
        needs radiation/wind data we don't have in the synthetic provider;
        here we use a temperature-driven approximation (Hargreaves-style)
        as a reasonable stand-in. When real regional weather CSVs are
        plugged in (with radiation/wind columns), this should be upgraded
        to full Penman-Monteith, matching the reference paper's Eq. 5."""
        temp = getattr(self.weather, "temp_c", None)
        if temp is not None:
            t_c = temp[min(t, len(temp) - 1)]
        else:
            t_c = 28.0  # fallback constant
        return max(0.0008 * (t_c + 17.8) * 8.0, 0.5)  # crude but bounded, mm/day

    def _get_state(self):
        stage_idx = min(self.t, len(self.stage_schedule) - 1)
        _, _, stage_name = self.stage_schedule[stage_idx]
        forecast = self.weather.get_forecast(self.t, config.FORECAST_HORIZON_DAYS)
        crop_condition = self.crop_condition_provider(self.t)
        state = np.concatenate([
            forecast,
            [self.h, config.H_MIN_MM, config.H_MAX_MM, config.H_P_MM, crop_condition],
        ]).astype(np.float32)
        return state

    def step(self, action_idx):
        action_frac = config.ACTIONS[action_idx]
        stage_idx = min(self.t, len(self.stage_schedule) - 1)
        kc, ky, stage_name = self.stage_schedule[stage_idx]

        # --- irrigation amount (Eq. 12 style) ---
        quota = config.H_MAX_MM - self.h
        irrigation_mm = action_frac * max(quota, 0.0)
        h_after_irrigation = self.h + irrigation_mm

        # --- actual weather / ET for today ---
        rainfall_today = self.weather.get_rainfall(self.t)
        et0 = self._reference_et0(self.t)
        ks = 1.0 if self.h >= config.RAW else max(self.h / max(config.RAW, 1e-6), 0.0)
        etc = kc * ks * et0          # actual crop evapotranspiration (Eq. 4 style)
        etm = kc * et0               # max (non-stressed) crop evapotranspiration

        # --- water balance update (Eq. 14 style) ---
        drainage = 0.0
        h_candidate = h_after_irrigation + rainfall_today - etc
        if h_candidate > config.H_P_MM:
            drainage = h_candidate - config.H_P_MM
            h_next = config.H_P_MM
        else:
            h_next = max(h_candidate, 0.0)

        crop_condition = self.crop_condition_provider(self.t)

        # --- reward terms ---
        r0 = self._basic_reward(action_idx, self.h)
        r1 = self._rainfall_utilization_reward(irrigation_mm, rainfall_today, drainage, etc)
        r2 = (etc / etm) ** ky if etm > 0 else 1.0
        r3 = config.STRESS_PENALTY[crop_condition] if action_idx == 0 else 1.0
        reward = r0 * r1 * r2 * r3

        self.cumulative_irrigation_mm += irrigation_mm
        self.cumulative_drainage_mm += drainage
        self.cumulative_actual_et += etc
        self.cumulative_max_et += etm

        self.h = h_next
        self.t += 1
        terminated = self.t >= config.SEASON_LENGTH_DAYS
        truncated = self.t >= len(self.weather)

        info = {
            "irrigation_mm": irrigation_mm, "drainage_mm": drainage,
            "r0": r0, "r1": r1, "r2": r2, "r3": r3, "stage": stage_name,
        }
        next_state = self._get_state() if not (terminated or truncated) else np.zeros(config.STATE_DIM, dtype=np.float32)
        return next_state, reward, terminated, truncated, info

    def _basic_reward(self, action_idx, h_before):
        """Mirrors reference paper Eq. 18: rewards conforming to the
        traditional 'irrigate only when below h_min' baseline, with a
        smaller reward for the cautious (50%) action than the full (100%)
        action when irrigation was actually needed."""
        below_min = h_before < config.H_MIN_MM
        if action_idx == 0:
            return 1.0 if not below_min else 1.0   # not violating baseline if above min; small reward if premature action avoided
        if action_idx == 1:
            return 9.0 if below_min else 1.0
        if action_idx == 2:
            return 10.0 if below_min else 1.0
        return 1.0

    def _rainfall_utilization_reward(self, irrigation_mm, rainfall_today, drainage, etc):
        denom = rainfall_today + irrigation_mm
        if denom <= 1e-6:
            return 1.0
        return max(1.0 - drainage / denom, 0.0) if drainage > 0 else 1.0
