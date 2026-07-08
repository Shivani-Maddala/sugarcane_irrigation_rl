"""
Integration layer: this is the literal implementation of the Step 2
architecture diagram's data flow --

    Leaf image -> CNN classifier -> crop condition class
                                          |
                                          v
    Weather/soil data -----------> State vector s_t -> DQN agent -> action

The CropConditionProvider class below is what the RL environment calls (as
`crop_condition_provider(t)`) at every timestep. It hides all the CNN
plumbing (image loading, preprocessing, inference) behind one simple
interface, so environment.py never needs to know a CNN exists -- exactly
the "keep pipelines independently testable" decision from Step 2.
"""

import os
import torch
from PIL import Image
from torchvision import transforms

from cnn import config as cnn_config
from cnn.model import build_model


class CropConditionProvider:
    """
    Wraps a trained CNN checkpoint + a day-indexed schedule of leaf images,
    and exposes __call__(t) -> int in {0, 1, 2} (healthy/moderate/severe),
    which is exactly what SugarcaneIrrigationEnv expects for its
    `crop_condition_provider` argument.

    image_schedule: dict mapping day index t -> path to a leaf image taken
    on that day. In a real deployment this would come from a field camera
    upload cadence (e.g. one photo per day); for now the caller supplies it
    explicitly so the RL side stays decoupled from *how* images arrive.

    If image_schedule has no entry for a given day (e.g. camera was offline),
    the provider reuses the most recent known condition rather than
    guessing -- a deliberate fail-safe rather than defaulting to "healthy",
    which could mask a real problem.
    """

    def __init__(self, checkpoint_path, image_schedule, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pretrained=False is correct here (not just an offline workaround):
        # we immediately overwrite every weight with the trained checkpoint
        # below, so downloading fresh ImageNet weights first would be wasted
        # bandwidth and a needless internet dependency at inference time.
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.image_schedule = image_schedule
        self.transform = transforms.Compose([
            transforms.Resize((cnn_config.IMAGE_SIZE, cnn_config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._last_known_condition = 0  # default assumption only for the very first day, if no image yet
        self._cache = {}

    def _classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            predicted_idx = int(logits.argmax(dim=1).item())
        return predicted_idx  # already 0=healthy, 1=moderate_stress, 2=severe_stress per cnn_config.CONDITION_CLASSES order

    def __call__(self, t):
        if t in self._cache:
            return self._cache[t]

        image_path = self.image_schedule.get(t)
        if image_path is None or not os.path.exists(image_path):
            condition = self._last_known_condition
        else:
            condition = self._classify_image(image_path)
            self._last_known_condition = condition

        self._cache[t] = condition
        return condition


class MockCropConditionProvider:
    """FOR INTEGRATION SMOKE-TESTING ONLY. Returns a fixed or randomly
    varying condition without touching the CNN at all, so the RL <-> CNN
    wiring can be verified even before a trained CNN checkpoint exists."""

    def __init__(self, sequence=None, cycle=True):
        self.sequence = sequence or [0]
        self.cycle = cycle

    def __call__(self, t):
        if self.cycle:
            return self.sequence[t % len(self.sequence)]
        return self.sequence[min(t, len(self.sequence) - 1)]
