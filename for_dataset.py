import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
import torch.nn.functional as nnf

from model import Net0, Net2

IMAGE_SIZE = (224, 224)
THRESH = 100
class_color = {0 : (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
class_name = {0 : 'noise', 1: 'person', 2: 'car'}

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(IMAGE_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

bgr = cv2.imread('bgr3.jpg')
grayBgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('cctv3.mp4')
cnt = 0

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cnt += 1
    print(cnt)
    if cnt < 1350 or cnt > 1370 or cnt % 2 != 0:
        continue
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(grayFrame, grayBgr)
    ret, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # dark = mask.copy()
    # dark = (dark > 300).astype(np.uint8)
    # cv2.drawContours(dark, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 0)
    right_border = frame.shape[0]
    down_border = frame.shape[1]

    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contour_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contour_poly)

        if boundRect[2] < 5 or boundRect[3] < 5:
            continue
        
        # cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), \
        # (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0, 255, 0), 2)
        
        pp = 15
        left = int(boundRect[1]) - pp
        right = int(boundRect[1]+boundRect[3]) + pp
        up = int(boundRect[0]) - pp
        down = int(boundRect[0]+boundRect[2]) + pp
        
        if left < 0:
            left += pp
        if right >= right_border:
            right -= pp
        if up < 0:
            up += pp
        if down >= down_border:
            down -= pp
        
        img = frame[left:right, up:down, :]
        if (img.shape[0] == 0 or img.shape[1] == 0):
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        cv2.imwrite(f'images/screenfromcctv3_2-{cnt}-{i}.png', img)

        
        

cv2.destroyAllWindows()
cap.release()
