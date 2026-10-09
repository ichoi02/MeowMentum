"""Stage 3: export the staged student (policies/student_policy<suffix>.pth) to the
single-file ONNX model the robot runs (policies/cat_controller<suffix>.onnx).

    python onnx_conversion.py --variant tail
    python onnx_conversion.py --variant notail

With torch 2.10.0 / onnx 1.21.0 / onnxscript 0.6.2 this reproduces the released ONNX
files byte for byte.
"""
import os
import torch
from variants import VARIANTS, policy_dir, student_path, onnx_path
# Imported rather than restated so the export cannot drift from the student.
# Student obs = [front_proj_grav(3), joint_angles(4), prev action(3), step(1)].
from distillation import StudentPolicy, STUDENT_OBS_DIM, ACT_DIM

STATE_DIM = STUDENT_OBS_DIM
ACTION_DIM = ACT_DIM

def main(variant="tail"):
    policy_dir()                    # policies/, created if missing
    PTH_FILE = student_path(variant)
    ONNX_FILE = onnx_path(variant)

    # Load the trained weights (a state_dict) into the student architecture.
    model = StudentPolicy(STATE_DIM, ACTION_DIM)
    try:
        model.load_state_dict(torch.load(PTH_FILE, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure you are loading a state_dict and the architecture matches.")
        return

    model.eval()

    # Dummy input used only to trace the graph. Shape (batch, STATE_DIM); the batch
    # axis is exported as dynamic.
    dummy_input = torch.randn(1, STATE_DIM, requires_grad=True)

    # Export at opset 14, then relabel the opset as 11 below: the graph is only
    # Gemm/Relu/Tanh, whose float32 semantics are the same at opset 11.
    print(f"Exporting model to {ONNX_FILE} as opset 14...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FILE,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print("PyTorch export complete. Now patching opset metadata...")

    import onnx

    # Load the newly exported model (pulling in any external tensor data)
    onnx_model = onnx.load(ONNX_FILE, load_external_data=True)

    # Force the metadata to say opset 11
    for imp in onnx_model.opset_import:
        if imp.domain == "" or imp.domain == "ai.onnx":
            imp.version = 11

    # Overwrite as a SINGLE self-contained file (no .onnx.data sidecar) so the Pi
    # only needs to copy one file.
    onnx.save(onnx_model, ONNX_FILE, save_as_external_data=False)
    sidecar = ONNX_FILE + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    print(f"Success! {ONNX_FILE} has been safely patched to Opset 11 (single file).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                         help="ablation condition to export (default: tail)")
    args = parser.parse_args()
    main(args.variant)
