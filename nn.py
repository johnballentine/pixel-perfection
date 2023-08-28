import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import glob

class NNDownscale(nn.Module):
    def __init__(self):
        super(NNDownscale, self).__init__()
        
        # Input shape: [batch, 4, 256, 256] (RGBA channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 64, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 64, 128, 128]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 128, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 128, 64, 64]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 256, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 256, 32, 32]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 512, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 512, 16, 16]
            
            nn.Conv2d(512, 4, kernel_size=1, stride=1, padding=0),  # Output shape: [batch, 4, 16, 16] (RGBA channels)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    

def train_model(directory, epochs, model_save_path="model.pth", batch_size=16):
    print(f"Training model from directory: {directory}")  # Debug print
    transform = transforms.ToTensor()
    label_files = sorted(glob.glob(os.path.join(directory, '*_label.png')))
    processed_files = sorted(glob.glob(os.path.join(directory, '*_processed.png')))
    
    print("Loading training data.")  # Debug print
    labels = [transform(Image.open(f)) for f in label_files]
    processed = [transform(Image.open(f)) for f in processed_files]

    labels = torch.stack(labels)
    processed = torch.stack(processed)
    
    print("Creating DataLoader.")  # Debug print
    dataset = TensorDataset(processed, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing loss and optimizer.")  # Debug print
    model = NNDownscale()  # No arguments needed
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    print("Training started.")  # Debug print
    temp_model_save_path = f"{model_save_path}_temp.pth"  # Temporary model file path
    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize epoch_loss to accumulate loss over the epoch

        for batch_idx, (batch_processed, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_processed)

            loss = criterion(outputs, batch_labels)
            epoch_loss += loss.item()  # Accumulate batch loss into epoch loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(dataloader)}")

            # Print the mean, median, min, and max for all channels
            all_channel_values = outputs.view(-1)  # Reshape tensor to combine all channel values

            mean_value = torch.mean(all_channel_values)
            median_value = torch.median(all_channel_values)
            min_value = torch.min(all_channel_values)
            max_value = torch.max(all_channel_values)

            print()
            print(f"Stat values:")
            print(f"  Mean: {mean_value}")
            print(f"  Median: {median_value}")
            print(f"  Min: {min_value}")
            print(f"  Max: {max_value}")
            print()
            print(f"Loss: {loss.item()}")
            print("----")

        avg_epoch_loss = epoch_loss / len(dataloader)  # Calculate average epoch loss
        scheduler.step(avg_epoch_loss)  # Step the scheduler based on the average epoch loss

        print(f"Average loss for Epoch {epoch+1}: {avg_epoch_loss}")
        print()

        # Save the model to temp after epoch to enable early stopping with lower risk of corruption
        torch.save(model.state_dict(), temp_model_save_path)
        os.rename(temp_model_save_path, model_save_path)
        print(f"Model copied to {model_save_path}")

def use_inference(model_path, input_image_path, output_image_path):
    print("Starting inference.")  # Debug print
    input_image = Image.open(input_image_path)
    print(f"Input image dimensions: {input_image.size}")  # Debug print
    channels = len(input_image.getbands())
    
    print(f"Input image has {channels} channels.")  # Debug print
    model = NNDownscale()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Loading and transforming input image.")  # Debug print
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0)
    
    print("Running inference.")  # Debug print
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(f"Output tensor shape: {output_tensor.shape}")

    # Generate unique filename
    counter = 0
    unique_output_path = output_image_path
    while os.path.exists(unique_output_path):
        counter += 1
        filename, file_extension = os.path.splitext(output_image_path)
        unique_output_path = f"{filename}_{counter}{file_extension}"

    # Rescale to 0-255 and convert to byte tensor
    output_tensor = (output_tensor.squeeze() * 255).byte()

    # Convert to PIL Image
    output_image = Image.fromarray(output_tensor.permute(1, 2, 0).numpy(), 'RGBA')

    
    print("Output tensor after squeeze shape:", output_tensor.shape)  # Debug print
    print("Output image size:", output_image.size)  # Debug print
    output_image.save(unique_output_path)

    print(f"Output image saved at {unique_output_path}")

def main(args):
    print("Parsing arguments.")  # Debug print
    if args.train:
        train_model(args.directory, args.epochs)
    elif args.infer:
        use_inference(args.model_path, args.input_image, args.output_image)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Art Cleaner")

    parser.add_argument('--directory', type=str, default='', help='Directory for training data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Use the model for inference')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model')
    parser.add_argument('--input_image', type=str, default='', help='Path to the input image for inference')
    parser.add_argument('--output_image', type=str, default='', help='Path to save the output image')
    
    args = parser.parse_args()
    print("Arguments parsed.")  # Debug print

    if args.train:
        train_model(args.directory, args.epochs, args.model_path)
    elif args.infer:
        use_inference(args.model_path, args.input_image, args.output_image)
    else:
        print("Invalid choice. Exiting.")
