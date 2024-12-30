import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def prewitt_operator(image):
    # Define Prewitt kernels
    kernel_x = torch.tensor([[[[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]],

                             [[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]],

                             [[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]]
                             ]], dtype=torch.float32)

    kernel_y = torch.tensor([[
        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]],

        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]]
    ]], dtype=torch.float32)

    # Set requires_grad to False to avoid gradients
    kernel_x = kernel_x.detach()
    kernel_y = kernel_y.detach()

    # Apply convolution
    edges_x = F.conv2d(image, kernel_x, stride=1, padding=1)
    edges_y = F.conv2d(image, kernel_y, stride=1, padding=1)

    # Calculate edge magnitude
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


# Load an image
image_path = '/home/ycren/python/testpic/Snipaste_2024-12-29_17-41-06.png'  # Replace with your image path
image = Image.open(image_path).convert('RGB')

# Transform the image to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Apply Prewitt operator
edges = prewitt_operator(image_tensor)

# Convert edges to numpy for visualization
edges_numpy = edges.squeeze(0).permute(1, 2, 0).detach().numpy()

# Display the original image and the edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edges Detected (Prewitt)')
plt.imshow(edges_numpy)
plt.axis('off')

plt.show()
