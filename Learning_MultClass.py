import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Helper function to find image file by ID
def find_image_path(directory, prefix, image_id):
    path = os.path.join(directory, f"{prefix}_id{image_id}.png")
    return path if os.path.exists(path) else None

# Custom Dataset class for Satellite and Building Masks
class SatelliteBuildingDataset(Dataset):
    def __init__(self, satellite_dir, heatmap_dir, transform=None, grey_threshold=10):
        self.satellite_dir = satellite_dir
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        self.grey_threshold = grey_threshold
        self.satellite_ids = [
            os.path.splitext(f)[0].split('_id')[-1] 
            for f in os.listdir(satellite_dir) 
            if os.path.isfile(os.path.join(satellite_dir, f)) and
               find_image_path(heatmap_dir, "heatmap", os.path.splitext(f)[0].split('_id')[-1]) is not None
        ]

    def __len__(self):
        return len(self.satellite_ids)

    def __getitem__(self, idx):
        image_id = self.satellite_ids[idx]
        satellite_path = find_image_path(self.satellite_dir, "satellite", image_id)
        heatmap_path = find_image_path(self.heatmap_dir, "heatmap", image_id)
        if satellite_path is None or heatmap_path is None:
            return None

        # Load images
        satellite_image = Image.open(satellite_path).convert("RGB")
        heatmap_image = Image.open(heatmap_path).convert("RGB")

        # Convert heatmap to binary mask
        heatmap_array = np.array(heatmap_image)
        building_mask = self.create_building_mask(heatmap_array)

        if self.transform:
            satellite_image = self.transform(satellite_image)
            building_mask = torch.tensor(building_mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return satellite_image, building_mask

    def create_building_mask(self, heatmap_array):
        # Create a binary mask where 1 = building (grey) and 0 = background
        grey_mask = np.std(heatmap_array, axis=-1) < self.grey_threshold
        return grey_mask.astype(np.float32)

# Custom collate function to filter out None values
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.default_collate(batch)

# Directories containing satellite and heatmap images
satellite_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Satellite'
heatmap_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Heatmap'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize the dataset
dataset = SatelliteBuildingDataset(satellite_dir, heatmap_dir, transform=transform)

# Split the dataset into training and validation sets
train_ids, val_ids = train_test_split(dataset.satellite_ids, test_size=0.2, random_state=42)
train_dataset = SatelliteBuildingDataset(satellite_dir, heatmap_dir, transform=transform, grey_threshold=10)
val_dataset = SatelliteBuildingDataset(satellite_dir, heatmap_dir, transform=transform, grey_threshold=10)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Decoder
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # Output single-channel for binary mask

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.enc3(F.max_pool2d(x2, 2)))
        # Decoder
        x = F.relu(self.dec1(F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)))
        x = F.relu(self.dec2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)))
        return torch.sigmoid(self.final(x))  # Apply sigmoid for binary classification

# Initialize model, loss, and optimizer
model = UNet()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary mask prediction
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop with visualization and progress tracking
def train_model(model, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Initialize the progress bar for training
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for satellite_images, masks in train_loader:
                optimizer.zero_grad()
                outputs = model(satellite_images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.update(1)  # Update the progress bar
                pbar.set_postfix(loss=loss.item())  # Display the current loss

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for satellite_images, masks in val_loader:
                outputs = model(satellite_images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")

    # Plot training and validation losses after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, num_epochs=50)
    torch.save(model.state_dict(), "building_mask_unet.pth")

