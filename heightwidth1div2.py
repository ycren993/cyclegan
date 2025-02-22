import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Read the image
image_path = '/home/ycren/python/DEBUG_CYCLEGAN/trainA/000001_jpg.rf.997bc55c23f00989c715fef593cedc59.jpg'  # Replace with your image path
image = Image.open(image_path)

# Convert the image to a PyTorch tensor (Image A)
transform_to_tensor = transforms.ToTensor()
image_A_tensor = transform_to_tensor(image)

# Resize the image to half its dimensions (Image B)
transform_resize = transforms.Resize((image_A_tensor.shape[1] // 2, image_A_tensor.shape[2] // 2))
image_B_tensor = transform_resize(image_A_tensor)

# Convert tensors to NumPy arrays for Matplotlib
image_A = image_A_tensor.numpy()
image_B = image_B_tensor.numpy()

# Transpose axes to match Matplotlib format (C, H, W) -> (H, W, C)
image_A = image_A.transpose(1, 2, 0)
image_B = image_B.transpose(1, 2, 0)

# Plot both images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_A)
plt.title('Image A (Original)')
plt.axis('off')  # Hide axes

plt.subplot(1, 2, 2)
plt.imshow(image_B)
plt.title('Image B (Resized)')
plt.axis('off')  # Hide axes

plt.tight_layout()
plt.show()