import cv2

def inter_p(box1, box2):
    if box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]:
        return 0
    
    if not((box1[0] > box2[0] and box1[1] < box2[1]) or (box1[2] > box2[2] and box1[3] < box2[3])):
        return 0
    
    left = max(box1[0], box2[0])
    right = min(box1[1], box2[1])
    up = max(box1[2], box2[2])
    down = min(box1[3], box2[3])
    p = ((right - left) * (down - up)) / ((box1[1] - box1[0]) * (box1[3] - box1[2]))
    if p > 1:
        p = 0
    return p

def find_boxes(frame, contours, PP = 0, MIN_SIZE = 0):
    right_border = frame.shape[0]
    down_border = frame.shape[1]
    boxes = []
    for _, c in enumerate(contours):
        contour_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contour_poly)

        if boundRect[2] < MIN_SIZE or boundRect[3] < MIN_SIZE:
            continue
        
        left = int(boundRect[1]) - PP
        right = int(boundRect[1]+boundRect[3]) + PP
        up = int(boundRect[0]) - PP
        down = int(boundRect[0]+boundRect[2]) + PP
            
        if left < 0:
            left += PP
        if right >= right_border:
            right -= PP
        if up < 0:
            up += PP
        if down >= down_border:
            down -= PP
            
        img = frame[left:right, up:down, :]
        if (img.shape[0] == 0 or img.shape[1] == 0):
            continue
            
        boxes.append([left, right, up, down, img, boundRect])

    return boxes

