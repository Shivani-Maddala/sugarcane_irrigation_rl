"""
Configuration for the sugarcane leaf-condition CNN.

IMPORTANT: RAW_TO_CONDITION below maps the *raw* folder/class names that ship
with the sugarcane leaf disease dataset (e.g. Daphal & Koli, 2022 / Kaggle
mirrors of it) to the 3 collapsed condition levels our RL agent's state
vector actually needs.

The exact raw folder names differ slightly between dataset mirrors (some use
"RedRot", others "Red_Rot" or "red rot"). THIS MAPPING MUST BE VERIFIED AND
UPDATED against the real folder names once the dataset is downloaded -- do
not train against this file assuming the names are already correct.
"""

import os

# ---- Paths -------------------------------------------------------------
RAW_DATA_DIR = os.path.join("data", "raw", "sugarcane_leaves")   # ImageFolder-style: RAW_DATA_DIR/<raw_class_name>/*.jpg
PROCESSED_DIR = os.path.join("data", "processed")
CHECKPOINT_DIR = os.path.join("results", "cnn_checkpoints")

# ---- Class mapping (VERIFY against real dataset before real training) --
RAW_TO_CONDITION = {
    "Healthy":         "healthy",
    "RedRot":          "severe_stress",
    "Mosaic":          "moderate_stress",
    "Rust":            "moderate_stress",
    "Yellow":      "moderate_stress"
}

CONDITION_CLASSES = ["healthy", "moderate_stress", "severe_stress"]  # order = label index 0,1,2
NUM_CLASSES = len(CONDITION_CLASSES)

# ---- Image / training hyperparameters ----------------------------------
IMAGE_SIZE = 224          # standard input size for ResNet18
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
