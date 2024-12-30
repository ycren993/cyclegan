import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/ycren/python/testpic/Snipaste_2024-12-29_17-41-06.png')
d = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sp = d.shape
print(sp)
height = sp[0]
weight = sp[1]
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
dSobel = np.zeros((height, weight))
dSobelx = np.zeros((height, weight))
dSobely = np.zeros((height, weight))
Gx = np.zeros(d.shape)
Gy = np.zeros(d.shape)
for i in range(height - 2):
    for j in range(weight - 2):
        Gx[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sx))
Gy[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sy))
dSobel[i + 1, j + 1] = (Gx[i + 1, j + 1] * Gx[i + 1, j + 1] + Gy[i + 1, j + 1] * Gy[i + 1, j + 1]) ** 0.5
dSobelx[i + 1, j + 1] = np.sqrt(Gx[i + 1, j + 1])
dSobely[i + 1, j + 1] = np.sqrt(Gy[i + 1, j + 1])
cv2.imshow('a', img)
cv2.imshow('b', d)
cv2.imshow('c', np.uint8(dSobel))
cv2.imshow('d', np.uint8(dSobelx))
cv2.imshow('e', np.uint8(dSobely))
cv2.waitKey(0)
cv2.destroyAllWindows()
a = np.uint8(dSobel)
b = np.uint8(dSobelx)
c = np.uint8(dSobel)
img = img[:, :, ::-1]

plt.subplot(321), plt.imshow(img), plt.title('f')
plt.subplot(322), plt.imshow(d, cmap=plt.cm.gray), plt.title('g')
plt.subplot(323), plt.imshow(a, cmap=plt.cm.gray), plt.title('w')
plt.subplot(324), plt.imshow(b, cmap=plt.cm.gray), plt.title('q')
plt.subplot(325), plt.imshow(c, cmap=plt.cm.gray), plt.title('e')
plt.show()
