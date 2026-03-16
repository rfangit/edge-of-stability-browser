"""
Export a subset of MNIST as a compact binary file for browser use.
Downsampled to 14x14 (196 pixels per image).

File format:
  - 4 bytes: uint32 little-endian, number of images (N)
  - 4 bytes: uint32 little-endian, number of pixels per image (196)
  - N bytes: uint8 labels (0-9)
  - N * 196 bytes: uint8 pixel data (0-255), row-major

Total for 250 images: 8 + 250 + 49000 = 49258 bytes (~48KB)

Usage:
  python export_mnist.py

Produces: mnist_subset.bin
"""

import struct
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F


def export_mnist_subset(
    output_path='mnist_subset.bin',
    digits=list(range(10)),
    n_per_digit=25,
    resolution=14,
    seed=42
):
    # Download/load MNIST
    train_dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.ToTensor()
    )

    pixel_dim = resolution * resolution

    # Collect indices per digit
    all_labels = torch.tensor([label for _, label in train_dataset])

    rng = np.random.RandomState(seed)
    selected_indices = []
    selected_labels = []

    for digit in digits:
        digit_indices = torch.where(all_labels == digit)[0].numpy()
        rng.shuffle(digit_indices)
        chosen = digit_indices[:n_per_digit]
        selected_indices.extend(chosen.tolist())
        selected_labels.extend([digit] * len(chosen))

    N = len(selected_indices)
    print(f"Selected {N} images ({n_per_digit} per digit x {len(digits)} digits)")
    print(f"Resolution: {resolution}x{resolution} = {pixel_dim} pixels per image")

    # Extract pixel data as uint8, with downsampling
    pixels = np.zeros((N, pixel_dim), dtype=np.uint8)
    for i, idx in enumerate(selected_indices):
        img, _ = train_dataset[idx]  # (1, 28, 28) float in [0, 1]
        if resolution < 28:
            img = F.adaptive_avg_pool2d(img, (resolution, resolution))
        pixels[i] = (img.view(-1).numpy() * 255).clip(0, 255).astype(np.uint8)

    labels = np.array(selected_labels, dtype=np.uint8)

    # Write binary file
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<II', N, pixel_dim))
        f.write(labels.tobytes())
        f.write(pixels.tobytes())

    file_size = 8 + N + N * pixel_dim
    print(f"Wrote {output_path}: {file_size} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Header: 8 bytes")
    print(f"  Labels: {N} bytes")
    print(f"  Pixels: {N * pixel_dim} bytes")

    # Verify by reading back
    with open(output_path, 'rb') as f:
        n_read, dim_read = struct.unpack('<II', f.read(8))
        labels_read = np.frombuffer(f.read(n_read), dtype=np.uint8)
        pixels_read = np.frombuffer(f.read(n_read * dim_read), dtype=np.uint8).reshape(n_read, dim_read)

    assert n_read == N
    assert dim_read == pixel_dim
    assert np.array_equal(labels_read, labels)
    assert np.array_equal(pixels_read, pixels)
    print("Verification passed!")

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Digit {u}: {c} images")


if __name__ == '__main__':
    export_mnist_subset()
