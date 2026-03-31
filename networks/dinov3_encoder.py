import math
from typing import Any, Optional

import torch
import torch.nn as nn
import timm


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str, strict: bool = False):
    """
    Flexible checkpoint loader for local .pth checkpoints.
    Handles several common checkpoint formats.
    """
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format: checkpoint is not a dict.")

    candidate_keys = ["state_dict", "model", "teacher", "student", "module", "network"]
    state_dict = None

    # case 1: checkpoint is directly a state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        for k in candidate_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break

    if state_dict is None:
        raise RuntimeError(f"Could not find usable state_dict. Keys found: {list(ckpt.keys())}")

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k

        # remove common wrappers
        changed = True
        while changed:
            changed = False
            for prefix in ["module.", "model.", "backbone.", "teacher.", "student."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    changed = True

        cleaned[new_k] = v

    incompatible = model.load_state_dict(cleaned, strict=strict)

    if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
    else:
        missing, unexpected = incompatible

    print(f"[INFO] strict={strict}")
    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[INFO] sample missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[INFO] sample unexpected keys:", unexpected[:20])

    return missing, unexpected


class DINOv3Backbone(nn.Module):
    """
    DINOv3 ViT backbone for segmentation.
    Uses timm and returns a spatial feature map: (B, C, Ht, Wt)

    Typical example for 224x224 input and patch16:
        input  : (B, 3, 224, 224)
        output : (B, C, 14, 14)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        weights_path: Optional[str] = None,
        pretrained: bool = False,
        freeze: bool = False,
        remove_cls_token: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.weights_path = weights_path
        self.pretrained = pretrained
        self.freeze = freeze
        self.remove_cls_token = remove_cls_token
        self.verbose = verbose

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if weights_path is not None and len(weights_path) > 0:
            load_checkpoint_flexible(self.backbone, weights_path, strict=False)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Could not infer backbone.num_features")

        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", None)

        if self.verbose:
            print(f"[INFO] model_name        : {self.model_name}")
            print(f"[INFO] embed_dim         : {self.embed_dim}")
            print(f"[INFO] num_prefix_tokens : {self.num_prefix_tokens}")

    def _extract_tensor_from_forward_features(self, feats: Any) -> torch.Tensor:
        """
        Handle common timm forward_features outputs.
        """
        if isinstance(feats, dict):
            preferred_keys = [
                "x_norm_patchtokens",   # ideal case: already only patch tokens
                "patch_tokens",
                "x_prenorm",
                "tokens",
                "features",
                "last_hidden_state",
            ]
            for key in preferred_keys:
                if key in feats and isinstance(feats[key], torch.Tensor):
                    if self.verbose:
                        print(f"[DEBUG] using dict key: {key} with shape {tuple(feats[key].shape)}")
                    return feats[key]

            tensor_items = [(k, v) for k, v in feats.items() if isinstance(v, torch.Tensor)]
            if len(tensor_items) == 0:
                raise RuntimeError(f"Unsupported dict output keys: {list(feats.keys())}")

            k, v = tensor_items[0]
            if self.verbose:
                print(f"[DEBUG] fallback dict key: {k} with shape {tuple(v.shape)}")
            return v

        if isinstance(feats, (tuple, list)):
            tensor_vals = [v for v in feats if isinstance(v, torch.Tensor)]
            if len(tensor_vals) == 0:
                raise RuntimeError("Unsupported tuple/list output from forward_features")
            if self.verbose:
                print(f"[DEBUG] using tuple/list tensor with shape {tuple(tensor_vals[-1].shape)}")
            return tensor_vals[-1]

        if isinstance(feats, torch.Tensor):
            if self.verbose:
                print(f"[DEBUG] forward_features returned tensor shape {tuple(feats.shape)}")
            return feats

        raise RuntimeError(f"Unsupported forward_features output type: {type(feats)}")

    def _tokens_to_map(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor features into spatial feature map.

        Supports:
            - (B, C, H, W): already spatial
            - (B, N, C): token sequence

        Handles:
            - pure patch tokens: N = H*W
            - prefixed tokens:   N = num_prefix_tokens + H*W
        """
        if feats.ndim == 4:
            return feats

        if feats.ndim != 3:
            raise RuntimeError(f"Unsupported feature tensor shape: {tuple(feats.shape)}")

        B, N, C = feats.shape

        # case 1: already only patch tokens
        side = int(math.sqrt(N))
        if side * side == N:
            if self.verbose:
                print(f"[DEBUG] tokens already patch-only: N={N}, side={side}")
            feats = feats.transpose(1, 2).contiguous().view(B, C, side, side)
            return feats

        # case 2: remove prefix tokens according to timm backbone
        num_prefix = getattr(self.backbone, "num_prefix_tokens", None)

        if num_prefix is not None and N > num_prefix:
            n_patch = N - num_prefix
            side = int(math.sqrt(n_patch))
            if side * side == n_patch:
                if self.verbose:
                    print(f"[DEBUG] removing num_prefix_tokens={num_prefix}: N={N} -> n_patch={n_patch}, side={side}")
                feats = feats[:, num_prefix:, :]
                feats = feats.transpose(1, 2).contiguous().view(B, C, side, side)
                return feats

        # case 3: fallback if num_prefix_tokens is unavailable
        fallback_prefix_candidates = [1, 5]  # CLS only, or CLS + 4 register tokens
        for prefix in fallback_prefix_candidates:
            if N > prefix:
                n_patch = N - prefix
                side = int(math.sqrt(n_patch))
                if side * side == n_patch:
                    if self.verbose:
                        print(f"[DEBUG] fallback removing prefix={prefix}: N={N} -> n_patch={n_patch}, side={side}")
                    feats = feats[:, prefix:, :]
                    feats = feats.transpose(1, 2).contiguous().view(B, C, side, side)
                    return feats

        raise RuntimeError(
            f"Cannot reshape tokens into square map. Got N={N}. "
            f"num_prefix_tokens={num_prefix}. "
            f"Expected N=H*W or N=prefix+H*W."
        )

    def forward_features_raw(self, x: torch.Tensor) -> Any:
        """
        Raw timm backbone forward_features output.
        """
        return self.backbone.forward_features(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw tensor extracted from forward_features.
        Shape is typically:
            - (B, N, C) for ViT tokens
            - or (B, C, H, W) for spatial output
        """
        raw = self.backbone.forward_features(x)

        if self.verbose:
            print(f"[DEBUG] raw output type: {type(raw)}")
            if isinstance(raw, dict):
                print(f"[DEBUG] raw dict keys: {list(raw.keys())}")

        feats = self._extract_tensor_from_forward_features(raw)

        if self.verbose and isinstance(feats, torch.Tensor):
            print(f"[DEBUG] extracted feats shape: {tuple(feats.shape)}")

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns spatial feature map for segmentation.
        Example:
            input  : (B, 3, 224, 224)
            output : (B, C, 14, 14)
        """
        feats = self.forward_tokens(x)
        fmap = self._tokens_to_map(feats)

        if self.verbose:
            print(f"[DEBUG] feature map shape: {tuple(fmap.shape)}")

        return fmap


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    model = DINOv3Backbone(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        weights_path="/data3/nkozah/my_project/Ibrahim_Dino_Unet/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        pretrained=False,
        freeze=False,
        remove_cls_token=True,
        verbose=True,
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 224, 224).to(device)
    print(f"[INFO] input shape: {tuple(x.shape)}")

    with torch.no_grad():
        raw = model.forward_features_raw(x)
        tokens = model.forward_tokens(x)
        fmap = model(x)

    print("\n[RESULTS]")
    print("raw type:", type(raw))
    if isinstance(raw, dict):
        print("raw keys:", list(raw.keys()))
    print("tokens shape:", tokens.shape if isinstance(tokens, torch.Tensor) else "not tensor")
    print("feature map shape:", fmap.shape)