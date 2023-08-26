import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import glob

class NNDownscale(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):  # Set channels to 4, considering the alpha channel
        super(NNDownscale, self).__init__()
        
        print("Initializing model.")  # Debug print
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, stride=2),  # Stride 2 for downscaling
            nn.ReLU(),
            nn.MaxPool2d(2),  # Further downscale
            
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # Stride 2 for downscaling
            nn.ReLU(),
            nn.MaxPool2d(2),  # Further downscale
            
            nn.Conv2d(128, out_channels, 1)  # 1x1 Conv to adjust channel number
        )

        print("Model initialized.")  # Debug print

    def forward(self, x):
        print("Running forward pass.")  # Debug print
        return self.encoder(x)


def train_model(directory, epochs, batch_size=16):
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
    model = NNDownscale(in_channels=processed.shape[1], out_channels=labels.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Training started.")  # Debug print
    for epoch in range(epochs):
        for batch_idx, (batch_processed, batch_labels) in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}.")  # Debug print
            optimizer.zero_grad()
            outputs = model(batch_processed)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}")

    print("Training complete.")  # Debug print
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved.")

def use_inference(model_path, input_image_path, output_image_path):
    print("Starting inference.")  # Debug print
    input_image = Image.open(input_image_path)
    channels = len(input_image.getbands())
    
    print(f"Input image has {channels} channels.")  # Debug print
    model = NNDownscale(channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Loading and transforming input image.")  # Debug print
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0)
    
    print("Running inference.")  # Debug print
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Generate unique filename
    counter = 0
    unique_output_path = output_image_path
    while os.path.exists(unique_output_path):
        counter += 1
        filename, file_extension = os.path.splitext(output_image_path)
        unique_output_path = f"{filename}_{counter}{file_extension}"

    output_image = transforms.ToPILImage()(output_tensor.squeeze())
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
    main(args)
