import cv2
import torch
import numpy as np

# Step 1: Read the image using OpenCV
image_path = '/home/ycren/python/EVUP_part/trainA/000016_jpg.rf.33c284fbfff2d837101100a62d3d1a2f.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Step 2: Convert the image from BGR to RGB (OpenCV loads images in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Convert the image to a PyTorch tensor
tensor_image = torch.tensor(image, dtype=torch.float32)

# Step 4: Optionally, normalize the tensor values to [0, 1]
tensor_image /= 255.0

# Step 5: Convert the tensor back to a NumPy array for saving
# Note: We need to convert it back to uint8 for saving as an image
numpy_image = (tensor_image.numpy() * 255).astype(np.uint8)

# Step 6: Save the image using OpenCV
cv2.imwrite('output_image.png', cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

print("Image saved as output_image.png")