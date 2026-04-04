"""
Export a policy to ONNX for hardware/controller.py.

Supports:
  - Stable-Baselines3 PPO checkpoints (.zip)
  - PyTorch .pth: StudentPolicy state_dict (see distillation.py) or a full nn.Module
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO


class OnnxableSB3Policy(torch.nn.Module):
    """
    Isolates the deterministic actor network from the SB3 PPO policy.
    This strips the value network and action sampling overhead.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


class StudentPolicy(nn.Module):
    """Must match distillation.py / test.py (MLP + Tanh)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def _infer_student_dims_from_state_dict(sd: dict) -> tuple[int, int]:
    """Infer (obs_dim, act_dim) from StudentPolicy Sequential weights."""
    if "net.0.weight" in sd and "net.4.weight" in sd:
        obs_dim = int(sd["net.0.weight"].shape[1])
        act_dim = int(sd["net.4.weight"].shape[0])
        return obs_dim, act_dim
    raise ValueError(
        "Cannot infer StudentPolicy shape from .pth; expected keys "
        "'net.0.weight' and 'net.4.weight'. Pass --student-obs-dim and --student-act-dim."
    )


def _load_pth(path: Path, obs_dim: int | None, act_dim: int | None) -> nn.Module:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        return ckpt

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported .pth content type: {type(ckpt)!r}")

    if obs_dim is None or act_dim is None:
        inferred_o, inferred_a = _infer_student_dims_from_state_dict(ckpt)
        obs_dim = obs_dim if obs_dim is not None else inferred_o
        act_dim = act_dim if act_dim is not None else inferred_a

    student = StudentPolicy(obs_dim, act_dim)
    student.load_state_dict(ckpt)
    student.eval()
    return student


def _export(
    model: nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
) -> None:
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["action"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=Path("cat_controller.zip"),
        help="PPO .zip or policy .pth (default: cat_controller.zip)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cat_controller.onnx"),
        help="Output ONNX path (default: cat_controller.onnx)",
    )
    parser.add_argument(
        "--student-obs-dim",
        type=int,
        default=None,
        help="Override student observation dim when loading .pth (optional if inferable)",
    )
    parser.add_argument(
        "--student-act-dim",
        type=int,
        default=None,
        help="Override student action dim when loading .pth (optional if inferable)",
    )
    args = parser.parse_args()

    model_path: Path = args.model
    onnx_path: Path = args.output

    suffix = model_path.suffix.lower()
    if suffix == ".zip":
        print(f"Loading SB3 model from {model_path}...")
        model = PPO.load(str(model_path), device="cpu")
        onnx_policy = OnnxableSB3Policy(model.policy)
        onnx_policy.eval()
        obs_shape = model.observation_space.shape
        dummy_input = torch.randn(1, *obs_shape)
        print(f"Exporting PPO policy to {onnx_path}...")
        _export(onnx_policy, dummy_input, onnx_path)

    elif suffix == ".pth":
        print(f"Loading PyTorch checkpoint from {model_path}...")
        net = _load_pth(model_path, args.student_obs_dim, args.student_act_dim)
        # Infer dummy shape from first parameter or forward input
        if isinstance(net, StudentPolicy):
            obs_dim = net.net[0].in_features
        else:
            found = False
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    obs_dim = m.in_features
                    found = True
                    break
            if not found:
                raise RuntimeError("Could not infer input size from loaded nn.Module")
        dummy_input = torch.randn(1, obs_dim)
        print(f"Exporting .pth policy to {onnx_path} (obs_dim={obs_dim})...")
        _export(net, dummy_input, onnx_path)

    else:
        raise SystemExit(f"Unsupported model extension {suffix!r}; use .zip or .pth")

    print("Export complete!")


if __name__ == "__main__":
    main()
