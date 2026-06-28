import os
import argparse
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import Grad-CAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import project-specific modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import build_model
import torchvision.transforms as VT
import torchaudio.transforms as AT

def get_args():
    parser = argparse.ArgumentParser(description="Visualize Model Attention (Grad-CAM)")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to sample watermelon skin image')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to sample watermelon audio/csv')
    parser.add_argument('--output_dir', type=str, default='./outputs/visuals_Grad_CAM', help='Directory to save heatmaps')
    return parser.parse_args()

def load_sample_data(img_path, config):
    # 1. Prepare skin image input
    img_size = config['audio'].get('img_size', [224, 224])
    
    # Load the original image for visualization (without preprocessing)
    raw_img = cv2.imread(img_path)
    raw_img = cv2.resize(raw_img, (img_size[0], img_size[1]))
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0 # Normalize image for Grad-CAM visualization

    # Prepare image tensor for PyTorch model input
    transform = VT.Compose([
        VT.Resize(tuple(img_size)),
        VT.ToTensor(),
        VT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0) # Add batch dimension

    # Provide a dummy mass value since we are only interested in image attention visualization

    mass_tensor = torch.tensor([[0.5]], dtype=torch.float32) 
    
    # The audio spectrogram should normally be generated as well. However, for simplicity, this script focuses on the skin image branch. A dummy audio tensor is supplied to satisfy the model's input requirements. In practice, image-based Grad-CAM visualizations are usually more informative.
    dummy_audio = torch.zeros(1, 3, img_size[0], img_size[1])

    return input_tensor, dummy_audio, mass_tensor, rgb_img

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load trained model
    print(f"Loading Model: {config['model']['name']}...")
    model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Select target layer for Grad-CAM
    # The target layer should correspond to the final convolutional block. Its location depends on the model architecture.

    target_layers = []
    if config['model']['name'] == 'fusion' or config['model']['name'] == 'cross':
        # For ResNet18-based Fusion/Cross models
        target_layers = [model.skin_net.layer4[-1]] 
    elif config['model']['name'] == 'convnext':
        # For ConvNeXt models
        target_layers = [model.skin_net.features[-1]]
    else:
        print("Grad-CAM is currently configured for Fusion/Cross (ResNet) or ConvNeXt.")
        return

    # 3. Load sample data
    print("Preparing Sample Data...")
    skin_tensor, audio_tensor, mass_tensor, rgb_img = load_sample_data(args.image_path, config)
    skin_tensor = skin_tensor.to(device)
    audio_tensor = audio_tensor.to(device)
    mass_tensor = mass_tensor.to(device)

    # 4. Create Grad-CAM object
    # For multi-input models, we create a temporary wrapper that allows Grad-CAM to focus only on the image branch while keeping the other inputs fixed.
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, audio, mass):
            super().__init__()
            self.model = model
            self.audio = audio
            self.mass = mass
        def forward(self, x):
            return self.model(self.audio, x, self.mass)

    cam_model = ModelWrapper(model, audio_tensor, mass_tensor)
    cam = GradCAM(model=cam_model, target_layers=target_layers)

    # 5. Generate attention heatmap

    print("Generating Heatmap...")
    # Leave targets=None so Grad-CAM automatically explains the model's predicted class (or regression output)
    grayscale_cam = cam(input_tensor=skin_tensor, targets=None)[0, :]

    # 6. Overlay heatmap on the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 7. Save and display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Skin Image")
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM (Model Attention)")
    plt.imshow(visualization)
    plt.axis('off')

    save_path = os.path.join(args.output_dir, "skin_attention.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Success! Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()