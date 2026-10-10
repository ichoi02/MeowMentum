"""The two robot variants (with and without a tail) and the paths of their policy files."""
import os

# Gymnasium environment and policy-file suffix of each variant
VARIANTS = {
    "tail":   {"env_id": "Cat-v0",       "suffix": ""},
    "notail": {"env_id": "CatNoTail-v0", "suffix": "_notail"},
}

# Directory of the staged policies, relative to the repository root
POLICY_DIR = "policies"


def policy_dir():
    """Create POLICY_DIR if it is missing and return it; call before writing a policy."""
    os.makedirs(POLICY_DIR, exist_ok=True)
    return POLICY_DIR


def teacher_path(variant):
    """Return the staged SAC teacher, policies/cat_controller<suffix>.zip."""
    return os.path.join(POLICY_DIR, f"cat_controller{VARIANTS[variant]['suffix']}.zip")


def student_path(variant):
    """Return the staged distilled student, policies/student_policy<suffix>.pth."""
    return os.path.join(POLICY_DIR, f"student_policy{VARIANTS[variant]['suffix']}.pth")


def onnx_path(variant):
    """Return the student exported for the robot, policies/cat_controller<suffix>.onnx."""
    return os.path.join(POLICY_DIR, f"cat_controller{VARIANTS[variant]['suffix']}.onnx")
