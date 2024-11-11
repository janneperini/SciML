import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
from Learning import UNet

# Load the trained model
model = UNet()
model.load_state_dict(torch.load("03TrainedModel/satellite_to_heatmap_unet.pth"))
model.eval()  # Set model to evaluation mode

# Directories
satellite_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Satellite'
heatmap_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Heatmap'
output_dir = '/Users/janne/Library/CloudStorage/OneDrive-ETHZurich/ML_Shared/Images/Prediction'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to predict and display heatmap for a random satellite image
def predict_and_display_heatmap(model, satellite_dir, heatmap_dir, output_dir, transform):
    satellite_files = os.listdir(satellite_dir)
    random_file = random.choice(satellite_files)
    
    # Load and transform the satellite image
    satellite_image_path = os.path.join(satellite_dir, random_file)
    satellite_image = Image.open(satellite_image_path)
    satellite_tensor = transform(satellite_image).unsqueeze(0)  # Add batch dimension

    # Load the corresponding training heatmap
    heatmap_image_path = os.path.join(heatmap_dir, random_file.replace("satellite", "heatmap"))
    heatmap_image = Image.open(heatmap_image_path)

    # Predict the heatmap
    with torch.no_grad():
        predicted_heatmap = model(satellite_tensor)

    # Remove batch dimension and convert to numpy
    predicted_heatmap = predicted_heatmap.squeeze(0).squeeze(0).cpu().numpy()

    # Save the predicted heatmap
    output_file = random_file.replace("satellite", "prediction")
    output_path = os.path.join(output_dir, output_file)
    save_image(torch.tensor(predicted_heatmap).unsqueeze(0), output_path)
    print(f"Predicted heatmap saved to {output_path}")

    # Display the satellite image, training heatmap, and predicted heatmap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(satellite_image)
    plt.title("Satellite Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_image)
    plt.title("Training Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_heatmap, cmap='viridis')  # Use 'viridis' colormap for heatmap
    plt.title("Predicted Heatmap")
    plt.axis("off")

    plt.show()

# Run the prediction and display
predict_and_display_heatmap(model, satellite_dir, heatmap_dir, output_dir, transform)
