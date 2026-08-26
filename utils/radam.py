"""
utils/radam.py

A RADAM-style feature extractor: pulls activation maps from several layers
of a pretrained ResNet50, projects each through a FIXED (untrained) random
linear projection, pools the result, and concatenates across layers into one
feature vector per image.

WHY THIS WORKS WITHOUT TRAINING THE CNN:
    A pretrained ImageNet backbone already encodes rich, general visual
    features (texture, color transitions, edges) even in a network that was
    never trained specifically for meat. Randomized projections capture a
    compressed sketch of the ACTIVATION PATTERNS themselves — statistically,
    a random projection preserves relative distances between different
    inputs' activations well enough (this is the same idea behind the
    Johnson-Lindenstrauss lemma) that a downstream classifier can still
    separate classes using it, without ever updating the CNN's weights.
    This is what the RADAM paper (Scabini et al.) calls "randomized
    aggregation of deep activation maps."

    Because nothing in the CNN is trained, this is far cheaper than
    fine-tuning — you run one forward pass per image, then train a small
    classifier (SVM) on the resulting feature vectors.

USAGE:
    from utils.radam import RadamFeatureExtractor
    extractor = RadamFeatureExtractor()          # builds fresh random projections
    extractor.save("models/radam_projections.pt") # save so inference matches training
    extractor = RadamFeatureExtractor.load("models/radam_projections.pt")  # reload later
    features = extractor.extract(pil_image)        # -> 1D numpy feature vector
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Which ResNet50 layers to pull activation maps from, and how many channels
# each one has (needed to build correctly-sized random projection matrices).
LAYER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}

PROJECTION_DIM = 32  # each layer's channels get randomly projected down to this size

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RadamFeatureExtractor:
    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.device = device
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.backbone.eval()
        self.backbone.to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False  # never trained — this is the whole point of RADAM

        self._activations = {}
        self._register_hooks()

        # Fixed random projection matrix per layer — created once, then
        # reused for every image, and saved/reloaded so extraction is
        # consistent between training and later inference.
        rng = torch.Generator().manual_seed(seed)
        self.projections = {
            name: torch.randn(channels, PROJECTION_DIM, generator=rng) / (channels ** 0.5)
            for name, channels in LAYER_CHANNELS.items()
        }

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook

        for name in LAYER_CHANNELS:
            layer = getattr(self.backbone, name)
            layer.register_forward_hook(make_hook(name))

    def extract(self, image: Image.Image) -> np.ndarray:
        """Runs one image through the backbone and returns a 1D feature vector."""
        input_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)

        self._activations = {}
        with torch.no_grad():
            self.backbone(input_tensor)

        layer_features = []
        for name in LAYER_CHANNELS:
            activation = self._activations[name]        # shape (1, C, H, W)
            b, c, h, w = activation.shape
            flat = activation.view(c, h * w).T            # (H*W, C) — one row per spatial location

            projected = flat @ self.projections[name]      # (H*W, PROJECTION_DIM)

            # Aggregate across all spatial locations with both mean and max —
            # mean captures the "typical" projected response, max captures
            # the strongest activation anywhere in the image for that layer.
            mean_pooled = projected.mean(dim=0)
            max_pooled = projected.max(dim=0).values

            layer_features.append(mean_pooled.numpy())
            layer_features.append(max_pooled.numpy())

        return np.concatenate(layer_features)  # final vector: 4 layers x 2 poolings x 32 dims = 256

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projections, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        extractor = cls(device=device)
        extractor.projections = torch.load(path, map_location=device)
        return extractor
