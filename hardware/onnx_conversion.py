"""Stage 3: export the staged student to the single-file ONNX model the robot runs.

    python -m hardware.onnx_conversion --variant tail      # policies/cat_controller.onnx
    python -m hardware.onnx_conversion --variant notail    # policies/cat_controller_notail.onnx

Reads policies/student_policy<suffix>.pth. With torch 2.10.0, onnx 1.21.0 and
onnxscript 0.6.2 this reproduces the released ONNX files byte for byte.
"""
import argparse
import os

import torch

from model.variants import VARIANTS, policy_dir, student_path, onnx_path
from train.distillation import StudentPolicy, STUDENT_OBS_DIM, ACT_DIM


def export(variant="tail"):
    """Export the staged student of a variant to policies/cat_controller<suffix>.onnx."""
    # Resolve the checkpoint and output paths, creating policies/ if it is missing
    policy_dir()
    pth_file = student_path(variant)
    onnx_file = onnx_path(variant)

    # Load the trained state_dict into the student network
    model = StudentPolicy(STUDENT_OBS_DIM, ACT_DIM)
    try:
        model.load_state_dict(torch.load(pth_file, map_location=torch.device("cpu")))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure you are loading a state_dict and the architecture matches.")
        return
    model.eval()

    # Export at opset 14, traced on a random (1, STUDENT_OBS_DIM) input with a dynamic batch axis
    dummy_input = torch.randn(1, STUDENT_OBS_DIM, requires_grad=True)
    print(f"Exporting model to {onnx_file} as opset 14...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    print("PyTorch export complete. Now patching opset metadata...")

    # Relabel the opset as 11; the Gemm/Relu/Tanh graph has the same float32 semantics there
    import onnx  # needed only from here, so --help and a failed weight load work without it
    onnx_model = onnx.load(onnx_file, load_external_data=True)
    for opset in onnx_model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 11

    # Save as one self-contained file, without a .onnx.data sidecar, for copying to the Pi
    onnx.save(onnx_model, onnx_file, save_as_external_data=False)
    sidecar = onnx_file + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"Success! {onnx_file} has been safely patched to Opset 11 (single file).")


def main():
    """Export the variant named on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                        help="ablation condition to export (default: tail)")
    args = parser.parse_args()
    export(args.variant)


if __name__ == "__main__":
    main()
