import torch
import torch.nn as nn
import time

# 1. Define your policy/model architecture. 
# This MUST exactly match the structure used during your RL training loop.
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() # Binds outputs to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# --- Configuration ---
# Update these dimensions to match your specific observation and action spaces
STATE_DIM = 22
ACTION_DIM = 3
PTH_FILE = "student_policy_notail.pth"
ONNX_FILE = f"cat_controller_{str(time.time())}.onnx"

def main():
    # 2. Instantiate the model and load the trained weights
    model = StudentPolicy(STATE_DIM, ACTION_DIM)
    
    # If you saved the full model (torch.save(model)), you would just load it directly.
    # If you saved the state_dict (recommended), load it like this:
    try:
        model.load_state_dict(torch.load(PTH_FILE, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure you are loading a state_dict and the architecture matches.")
        return

    # 3. Set the model to evaluation mode
    # Crucial for freezing layers like BatchNorm and Dropout so they behave deterministically
    model.eval()

    # 4. Create a dummy input tensor
    # This acts as a probe to trace the computational graph. 
    # Shape here is (Batch_Size=1, State_Dimension)
    dummy_input = torch.randn(1, STATE_DIM, requires_grad=True)

    # 5. Export to ONNX
    print(f"Exporting model to {ONNX_FILE}...")
    torch.onnx.export(
        model,                      # The instantiated PyTorch model
        dummy_input,                # The dummy input tuple/tensor
        ONNX_FILE,                  # Output file path
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=11,           # Opset 11 is highly stable for most ONNX Runtime environments
        do_constant_folding=True,   # Optimizes the graph by folding constants 
        input_names=['state'],      # Define a clear input name for the C++/Python inference session
        output_names=['action'],    # Define a clear output name
        
        # Optional: Define dynamic axes if you plan to pass variable batch sizes during inference.
        # If your embedded controller strictly evaluates one state at a time, you can omit this.
        dynamic_axes={
            'state': {0: 'batch_size'},    
            'action': {0: 'batch_size'}
        }
    )
    print("Conversion complete!")

if __name__ == "__main__":
    main()