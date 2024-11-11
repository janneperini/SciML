import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Helper function to find image file by ID
def find_image_path(directory, prefix, image_id):
    path = os.path.join(directory, f"{prefix}_id{image_id}.png")
    return path if os.path.exists(path) else None

# Custom Dataset class for Satellite and Heatmap images
class SatelliteHeatmapDataset(Dataset):
    def __init__(self, satellite_dir, heatmap_dir, transform=None):
        self.satellite_dir = satellite_dir
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        self.satellite_ids = [os.path.splitext(f)[0].split('_id')[-1] 
                              for f in os.listdir(satellite_dir) if os.path.isfile(os.path.join(satellite_dir, f))]

    def __len__(self):
        return len(self.satellite_ids)

    def __getitem__(self, idx):
        image_id = self.satellite_ids[idx]
        satellite_path = find_image_path(self.satellite_dir, "satellite", image_id)
        heatmap_path = find_image_path(self.heatmap_dir, "heatmap", image_id)
        if satellite_path is None or heatmap_path is None:
            print(f"File not found: {satellite_path} or {heatmap_path}")
            return None
        satellite_image = Image.open(satellite_path)
        heatmap_image = Image.open(heatmap_path)
        if self.transform:
            satellite_image = self.transform(satellite_image)
            heatmap_image = self.transform(heatmap_image)
        return satellite_image, heatmap_image

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
dataset = SatelliteHeatmapDataset(satellite_dir, heatmap_dir, transform=transform)

# Split the IDs into training and validation sets
train_ids, val_ids = train_test_split(dataset.satellite_ids, test_size=0.2, random_state=42)

class SplitSatelliteHeatmapDataset(Dataset):
    def __init__(self, satellite_dir, heatmap_dir, ids, transform=None):
        self.satellite_dir = satellite_dir
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        satellite_path = find_image_path(self.satellite_dir, "satellite", image_id)
        heatmap_path = find_image_path(self.heatmap_dir, "heatmap", image_id)
        if satellite_path is None or heatmap_path is None:
            print(f"File not found: {satellite_path} or {heatmap_path}")
            return None
        satellite_image = Image.open(satellite_path)
        heatmap_image = Image.open(heatmap_path)
        if self.transform:
            satellite_image = self.transform(satellite_image)
            heatmap_image = self.transform(heatmap_image)
        return satellite_image, heatmap_image

# Create separate datasets
train_dataset = SplitSatelliteHeatmapDataset(satellite_dir, heatmap_dir, train_ids, transform=transform)
val_dataset = SplitSatelliteHeatmapDataset(satellite_dir, heatmap_dir, val_ids, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define encoder layers
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Define decoder layers
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # Output single-channel heatmap

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.enc3(F.max_pool2d(x2, 2)))
        
        # Decoder
        x = F.relu(self.dec1(F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)))
        x = F.relu(self.dec2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)))
        
        return self.final(x)

# Initialize model
model = UNet()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop with graceful KeyboardInterrupt handling
def train_model(model, train_loader, val_loader, num_epochs=20):
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            
            for satellite_images, heatmap_images in train_loader:
                optimizer.zero_grad()
                outputs = model(satellite_images)
                loss = criterion(outputs, heatmap_images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

            # Evaluate on validation set
            val_loss = evaluate_model(model, val_loader)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss}")
    
    except KeyboardInterrupt:
        print("Training interrupted! Plotting current progress...")

    # Plot the training and validation loss even if interrupted
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

def evaluate_model(model, val_loader):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for satellite_images, heatmap_images in val_loader:
            outputs = model(satellite_images)
            loss = criterion(outputs, heatmap_images)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=20)

    # Save the trained model
    torch.save(model.state_dict(), "satellite_to_heatmap_unet.pth")
