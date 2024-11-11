import os
import random
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Learning_MultClass import UNetMultiClass

# Load the trained model
model = UNetMultiClass(num_classes=2)
model.load_state_dict(torch.load("satellite_to_heatmap_unet_multiclass.pth"))
model.eval()  # Set model to evaluation mode

# Directories
satellite_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Satellite'
output_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Prediction'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to predict heatmap for a random satellite image
def predict_and_save_heatmap(model, satellite_dir, output_dir, transform):
    satellite_files = os.listdir(satellite_dir)
    random_file = random.choice(satellite_files)
    
    # Load and transform the satellite image
    satellite_image = Image.open(os.path.join(satellite_dir, random_file))
    satellite_tensor = transform(satellite_image).unsqueeze(0)  # Add batch dimension

    # Predict the heatmap
    with torch.no_grad():
        output = model(satellite_tensor)
        predicted_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Save the predicted heatmap
    output_file = random_file.replace("satellite", "prediction")
    output_path = os.path.join(output_dir, output_file)
    Image.fromarray((predicted_classes * 255).astype(np.uint8)).save(output_path)
    print(f"Predicted heatmap saved to {output_path}")

    # Display the satellite image and the predicted heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(satellite_image)
    plt.title("Satellite Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_classes, cmap='viridis')  # Use 'viridis' colormap for heatmap
    plt.title("Predicted Heatmap")
    plt.axis("off")

    plt.show()

# Run the prediction and save the output
predict_and_save_heatmap(model, satellite_dir, output_dir, transform)
