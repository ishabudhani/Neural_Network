import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: Takes 3 channels (RGB), outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Layer 2: Takes 16 channels, outputs 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pooling layer to reduce spatial dimensions (width/height) by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers (Classifier)
        # Assuming input image size is 32x32, two rounds of pooling make it 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the 3D tensor into a 1D vector for the linear layers
        x = x.view(-1, 32 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = SimpleCNN(num_classes=10)
print(model)