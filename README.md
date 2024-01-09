Generate synthetic data to train a model to restore pixel perfection and background removal of web images of pixel art<br>
<br>
data - ground truth. there are much larger datasets and other resolutions.<br>
<br>
add_alpha.py - adds a blank alpha channel turning a BGR image into a BGRA image<br>
bg.py - background generator. generates a bunch of randomized background with perlin noise and more. i believe this is much larger but part got truncated by chatgpt. it will be in earlier commit<br>
image_info.py - fast tool for training data image diagnostics<br>
main.py - main logic but also a toolkit of functions for noisy synthetic data generation<br>
nn.py - the model architecture. inference and training is done using the CLI tools in here<br>
