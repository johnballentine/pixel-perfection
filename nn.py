import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import glob

class PixelCNN(nn.Module):
    def __init__(self, channels=128):
        super(PixelCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)  # 1x1 convolution to keep dimensions consistent
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(directory, epochs):
    transform = transforms.ToTensor()
    label_files = sorted(glob.glob(os.path.join(directory, '*_label.png')))
    processed_files = sorted(glob.glob(os.path.join(directory, '*_processed.png')))
    
    labels = [transform(Image.open(f)) for f in label_files]
    processed = [transform(Image.open(f)) for f in processed_files]

    labels = torch.stack(labels)
    processed = torch.stack(processed)

    # Initialize model, loss, and optimizer
    model = PixelCNN(processed.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(processed)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved.")

def use_inference(model_path, input_image_path, output_image_path):
    # Dynamically determine the number of channels
    input_image = Image.open(input_image_path)
    channels = len(input_image.getbands())  # This will return 3 for RGB and 4 for RGBA, for example
    
    model = PixelCNN(channels)  # Initialize model with the correct number of channels
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0)
    
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

    if args.train and not args.directory:
        print("Error: Directory must be specified for training.")
        exit(1)

    main(args)

