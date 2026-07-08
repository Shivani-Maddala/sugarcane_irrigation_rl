"""
Configuration for the sugarcane irrigation DQN environment and agent.

Growth-stage lengths and Kc/Ky values are sourced directly from FAO's Land &
Water sugarcane crop page (Kc/Ky table) -- NOT invented. We model a RATOON
crop under TROPICS stage lengths, since Maharashtra sugarcane farming is
predominantly ratoon-based and Maharashtra's climate maps to FAO's "Tropics"
column rather than "Hawaii" or generic "Low Latitudes."

If you'd rather model a virgin (first-planting) crop, or a different
latitude column, change GROWTH_STAGES below -- everything downstream reads
from this table, nothing is hardcoded elsewhere.
"""

# ---- Growth stages: (name, length_days, Kc, Ky) ------------------------
# Kc/Ky assigned per FAO stage; development stage interpolates Kc linearly
# between initial and mid (as FAO56 prescribes with ">>" in its table).
GROWTH_STAGES = [
    {"name": "initial",     "length_days": 30,  "kc": 0.40, "ky": 0.75},
    {"name": "development", "length_days": 50,  "kc": None, "ky": 0.75},  # Kc interpolated
    {"name": "mid_season",  "length_days": 180, "kc": 1.25, "ky": 0.50},
    {"name": "late_season", "length_days": 60,  "kc": 0.75, "ky": 0.10},
]
SEASON_LENGTH_DAYS = sum(s["length_days"] for s in GROWTH_STAGES)  # 320
KY_TOTAL = 1.2  # whole-season yield response factor, FAO value

# ---- Soil / water parameters (Maharashtra clay-loam-type assumption, ----
# ---- consistent with the reference paper's soil assumptions) -----------
DEPLETION_FRACTION_P = 0.65   # FAO value for sugarcane
ROOT_DEPTH_M = 1.5            # FAO value for sugarcane
THETA_FC_MINUS_WP = 0.16      # same clay-loam assumption the reference paper used (Sec 2.3)
TAW = 1000 * THETA_FC_MINUS_WP * ROOT_DEPTH_M   # total available water, mm
RAW = DEPLETION_FRACTION_P * TAW                # readily available water, mm

# Field water depth limits by growth-stage "phase" -- mirrors the reference
# paper's Table 2 (h_min, h_max, H_p per stage). Sugarcane is NOT flood
# irrigated like paddy rice, so these represent soil-moisture-equivalent
# depth bands (mm of available water in the root zone), not standing water.
H_MIN_MM = 0.30 * TAW    # irrigate when available water drops below this
H_MAX_MM = TAW           # field capacity equivalent
H_P_MM = TAW             # max allowable "depth" after rainfall before drainage/runoff

# ---- RL state / action space -------------------------------------------
FORECAST_HORIZON_DAYS = 7          # matches reference paper's 7-day forecast
CROP_CONDITION_LEVELS = 3          # healthy=0, moderate_stress=1, severe_stress=2 (from CNN)
# state = [P_t (7), h_t, h_min, h_max, H_p, crop_condition]  -> 12 dims
STATE_DIM = FORECAST_HORIZON_DAYS + 4 + 1
ACTIONS = [0.0, 0.5, 1.0]          # fraction of quota supplied: 0%, 50%, 100%
NUM_ACTIONS = len(ACTIONS)

# ---- DQN hyperparameters (following the reference paper's design) -------
GAMMA = 0.2               # discount factor, same value the reference paper used
LEARNING_RATE = 3e-4
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 50    # steps, i.e. "C" in the reference paper's Algorithm 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995        # per-episode multiplicative decay
NUM_EPISODES = 300           # one episode = one 320-day ratoon season

# ---- Reward function weights --------------------------------------------
# r_t = r0 * r1 * r2 * r3
# r0: conformity to traditional irrigation baseline (same logic as reference paper Eq. 18)
# r1: rainfall-utilization reward (Eq. 16 style)
# r2: yield reward via crop water production function (Eq. 17 style)
# r3: NEW -- crop-condition stress penalty from the CNN, penalizes under-irrigating
#     a visibly stressed/diseased crop
STRESS_PENALTY = {0: 1.0, 1: 0.85, 2: 0.6}  # r3 when action==0 (no irrigation) and condition is healthy/moderate/severe
