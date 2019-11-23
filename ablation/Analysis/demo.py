import cv2
import torch
import numpy as np
import torch.nn.functional as F


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


def main():
    img_path = 'cat1.jpg'
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    mask = torch.rand(14,14)
    mask = F.interpolate(mask,(224,224))
    show_cam_on_image(img, mask)

if __name__ == '__main__':
    main()