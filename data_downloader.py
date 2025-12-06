import os
import shutil

import numpy as np
from torchvision import datasets, transforms

save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Download and load the EMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.EMNIST(
    root="./data", split="balanced", train=True, download=True, transform=transform
)
mnist_test = datasets.EMNIST(
    root="./data", split="balanced", train=False, download=True, transform=transform
)

# Convert to numpy arrays and normalize
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = mnist_train.targets.numpy().astype(np.int32)
X_test = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = mnist_test.targets.numpy().astype(np.int32)

# Save the data as raw binary files
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

print("EMNIST dataset has been downloaded and saved in binary format in data/ directory.")
