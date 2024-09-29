import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Function to register hooks and capture gradients
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    # Forward hook to capture the activations
    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    # Backward hook to capture the gradients
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_tensor, class_idx=1):
        # Run forward pass
        output = self.model(img_tensor.unsqueeze(0))  # Add batch dimension

        # Backward pass to calculate gradients
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Get pooled gradients across channels
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations by the pooled gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_grads[i]

        # Compute the heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) != 0:  # Avoid divide by zero
            heatmap /= np.max(heatmap)  # Normalize
        return heatmap

# Utility function to overlay heatmap on the image
def overlay_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))  # Resize to match original image size
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, colormap)  # Apply color map

    img = img.permute(1, 2, 0).cpu().numpy()  # Change dimensions
    img = np.uint8(255 * (img - img.min()) / (img.max() - img.min()))  # Normalize image to [0, 255]

    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)  # Overlay using cv2.addWeighted
    return superimposed_img

# Function to visualize the Grad-CAM results
def visualize_gradcam(img_tensor, model, target_layer, class_idx=None, save_path=None):
    # Instantiate the Grad-CAM object
    gradcam = GradCAM(model, target_layer)

    # Generate the Grad-CAM heatmap
    heatmap = gradcam.generate_heatmap(img_tensor, class_idx)

    # Overlay the heatmap on the original image
    result = overlay_heatmap(heatmap, img_tensor)

    # Plot the original image and the overlay
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(result)
    ax[1].set_title('Grad-CAM')
    ax[1].axis('off')

    # plt.show()

    if save_path is not None:
        # Save the Grad-CAM image as a file
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*6*6, 2)
    def forward(self, img_batch):
        a = F.relu(self.conv1(img_batch))
        a = F.relu(self.conv2(a))
        a = self.pool1(a)
        a = F.relu(self.conv3(a))
        a = F.relu(self.conv4(a))
        a = self.pool2(a)
        a = F.relu(self.conv5(a))
        a = F.relu(self.conv6(a))
        a = self.pool3(a)
        a = torch.flatten(a)            # flatten all dimensions except batch
        a = self.fc(a)
        return a



def main():
    img_path = "test.png"        
    image = io.imread(img_path)
    ## Apply transformations to the image
    image = transforms.ToTensor()(image)

    # z-score normalization
    image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)

    model = Net3()
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu'), weights_only=True))

    prediction = model(image)
    prediction = prediction.tolist()

    print(prediction)

    # Usage Example:
    # Assume you are using Net2 model
    # model = Net3()
    target_layer = model.conv2 # You can choose any convolutional layer for Grad-CAM
    img_index = 15
    # Visualize Grad-CAM for the first image in the dataset
    visualize_gradcam(image, model, target_layer, save_path="gradcam_output.png")

    print(prediction)
    return prediction[0] < prediction[1]


if __name__ == "__main__":
    main()



