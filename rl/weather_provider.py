"""
Weather data provider for the sugarcane irrigation environment.

Design: the RL environment should not care whether its rainfall/ET data
comes from a real CSV (Kolhapur/Pune/Solapur weather records) or a
synthetic generator -- it just calls `get_day(t)` and `get_forecast(t)`.
This lets us smoke-test the environment now with synthetic data, then
swap in real regional CSVs later with ZERO changes to environment.py.
"""

import numpy as np
import pandas as pd


class WeatherDataProvider:
    """Loads a real CSV with (at minimum) columns:
    date, rainfall_mm, temp_c, humidity_pct, wind_speed_ms, sunshine_hours
    and an ET0 column if available (else ET0 is computed elsewhere).
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, parse_dates=["date"]).reset_index(drop=True)
        required = {"rainfall_mm", "temp_c", "humidity_pct"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Weather CSV is missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def get_rainfall(self, t):
        return float(self.df.loc[t, "rainfall_mm"])

    def get_forecast(self, t, horizon):
        """Naive 'forecast' = actual future rainfall (perfect-foresight upper
        bound) UNLESS the CSV has explicit forecast_1..forecast_7 columns,
        which we prefer if present (more realistic, matches reference paper's
        approach of using real, imperfect public forecasts)."""
        forecast_cols = [f"forecast_day{i}" for i in range(1, horizon + 1)]
        if all(c in self.df.columns for c in forecast_cols):
            return self.df.loc[t, forecast_cols].to_numpy(dtype=float)
        # Fallback: use actual future rainfall, clipped at series end with zeros.
        end = min(t + horizon, len(self.df))
        actual = self.df.loc[t:end - 1, "rainfall_mm"].to_numpy(dtype=float)
        if len(actual) < horizon:
            actual = np.concatenate([actual, np.zeros(horizon - len(actual))])
        return actual


class SyntheticWeatherProvider:
    """FOR SMOKE-TESTING ONLY. Generates plausible-looking but entirely
    synthetic daily rainfall/temperature/humidity so the environment code
    can be exercised without a real regional dataset. Do not use results
    from this provider as real findings."""

    def __init__(self, num_days, seed=0):
        rng = np.random.default_rng(seed)
        # Rough monsoon-like pattern: higher rain probability mid-season.
        day_idx = np.arange(num_days)
        rain_prob = 0.15 + 0.35 * np.sin(np.pi * day_idx / num_days) ** 2
        rains = rng.random(num_days) < rain_prob
        amounts = np.where(rains, rng.gamma(shape=2.0, scale=8.0, size=num_days), 0.0)
        self.rainfall = amounts
        self.temp_c = 24 + 6 * np.sin(2 * np.pi * day_idx / num_days) + rng.normal(0, 1, num_days)
        self.humidity_pct = 60 + 20 * np.sin(np.pi * day_idx / num_days) + rng.normal(0, 3, num_days)
        self.num_days = num_days

    def __len__(self):
        return self.num_days

    def get_rainfall(self, t):
        return float(self.rainfall[t])

    def get_forecast(self, t, horizon):
        end = min(t + horizon, self.num_days)
        actual = self.rainfall[t:end]
        if len(actual) < horizon:
            actual = np.concatenate([actual, np.zeros(horizon - len(actual))])
        # Add forecast noise so it's not perfect foresight (more realistic).
        noisy = actual * np.random.default_rng(t).uniform(0.6, 1.3, size=horizon)
        return noisy
