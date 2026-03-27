import torch
from stable_baselines3 import PPO

MODEL_PATH = "cat_controller.zip"
ONNX_PATH = "cat_controller.onnx"

class OnnxableSB3Policy(torch.nn.Module):
    """
    Isolates the deterministic actor network from the SB3 PPO policy.
    This strips the value network and action sampling overhead.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # Extract features (handles observation preprocessing)
        features = self.policy.extract_features(obs)
        # Pass through the actor's MLP layers
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        # Output the deterministic action
        return self.policy.action_net(latent_pi)

def main():
    print(f"Loading SB3 model from {MODEL_PATH}...")
    # Load model on CPU
    model = PPO.load(MODEL_PATH, device="cpu")
    
    # Wrap the policy
    onnx_policy = OnnxableSB3Policy(model.policy)
    onnx_policy.eval()

    # Create a dummy observation matching the environment's space
    obs_shape = model.observation_space.shape
    dummy_input = torch.randn(1, *obs_shape)

    print(f"Exporting policy to {ONNX_PATH}...")
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print("Export complete!")

if __name__ == "__main__":
    main()