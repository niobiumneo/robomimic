import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel
from robomimic.models.obs_core import EncoderCore


class DinoV2Core(EncoderCore):
    """
    Frozen DINOv2 encoder for robomimic image observations.
    Input shape should be [C, H, W].
    Output is a feature vector [D].
    """

    def __init__(
        self,
        input_shape,
        model_name="facebook/dinov2-base",
        feature_dimension=128,
        freeze_backbone=True,
        use_cls_token=True,
    ):
        super().__init__(input_shape=input_shape)

        assert len(input_shape) == 3, f"Expected input shape [C,H,W], got {input_shape}"
        assert input_shape[0] == 3, f"Expected 3-channel input, got {input_shape}"

        self.model_name = model_name
        self.feature_dimension = feature_dimension
        self.freeze_backbone = freeze_backbone
        self.use_cls_token = use_cls_token

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_dim = self.backbone.config.hidden_size
        self.image_size = int(self.backbone.config.image_size)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dimension),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "image_mean",
            torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def output_shape(self, input_shape=None):
        return [self.feature_dimension]

    def _preprocess(self, x):
        """
        Expect x as [B,C,H,W].
        robomimic usually uses channel-first for images.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {tuple(x.shape)}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0

        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )

        x = (x - self.image_mean) / self.image_std
        return x

    def forward(self, inputs):
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape), \
            f"Expected trailing shape {self.input_shape}, got {tuple(inputs.shape)}"

        x = self._preprocess(inputs)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
        else:
            outputs = self.backbone(pixel_values=x)

        tokens = outputs.last_hidden_state  # [B, 1+N, hidden_dim]

        if self.use_cls_token:
            feat = tokens[:, 0, :]
        else:
            feat = tokens[:, 1:, :].mean(dim=1)

        # z = self.projector(feat)
        # z = F.normalize(z, dim=-1)
        feat = tokens[:, 0, :]   # shape [B, 768] for dinov2-base
        z = F.normalize(feat, dim=-1)
        return z
    
x = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
enc = DinoV2Core(input_shape=(3, 224, 224))
z = enc(x)
print("Latent shape:", z.shape)