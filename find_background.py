import cv2
import matplotlib.pyplot as plt
import numpy as np

def move_bg(file_name = "cctv.mp4", max_frame = None, alpha = 0.95, thresh = 100):
    if max_frame is None:
        max_frame = -1

    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, frame = cap.read()
    cnt = 1
    bgr = frame.copy()
    prevGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret or cnt == max_frame:
            break

        cnt += 1
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(grayFrame, prevGrayFrame)

        mask = diff < thresh
        mask = np.array([mask, mask, mask])
        alpha_matrix = mask * alpha + (mask == 0)
        alpha_matrix.resize(alpha_matrix.shape[1], alpha_matrix.shape[2], alpha_matrix.shape[0])
        bgr = alpha_matrix * bgr + (1 - alpha_matrix) * frame
        bgr = bgr.astype(np.uint8)

        prevGrayFrame = grayFrame.copy()

        print(cnt)

    cap.release()
    return bgr


def mean_bg(file_name = "cctv.mp4", max_frame = None):
    if max_frame is None:
        max_frame = -1
    
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, frame = cap.read()

    cnt = 1
    cntMatrix = frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret or cnt == max_frame:
            break

        cnt += 1
        cntMatrix = (cntMatrix * (cnt - 1) + frame) / cnt
        print(cnt)

    cap.release()
    bgr = cntMatrix.astype(np.uint8)

    return bgr

def top_bg(file_name = "cctv.mp4", max_frame = None, check_cnt = 150):
    if max_frame is None:
        max_frame = -1
    
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, frame = cap.read()

    h = frame.shape[0]
    w = frame.shape[1]

    cnt = 1

    bgr = frame.copy()
    frames_saver = np.array([bgr])
    all_video = np.array([bgr])

    while True:
        ret, frame = cap.read()
        if not ret or cnt == max_frame:
            break

        cnt += 1
        frames_saver = np.append(frames_saver, np.array([frame]), axis = 0)

        if cnt % check_cnt == 0:
            for i in range(h):
                for j in range(w):
                    for c in range(3):
                        bgr[i, j, c] = np.bincount(frames_saver[:, i, j, c]).argmax()
            if cnt == check_cnt:
                all_video = np.array([bgr])
            else:
                all_video = np.append(all_video, np.array([bgr]), axis = 0)
            frames_saver = np.array([bgr])
        print(cnt)

    cap.release()
    for i in range(h):
        for j in range(w):
            for c in range(3):
                bgr[i, j, c] = np.bincount(all_video[:, i, j, c]).argmax()

    return bgr

def mean_bg(file_name = "cctv.mp4", max_frame = None):
    if max_frame is None:
        max_frame = -1
    
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, frame = cap.read()

    cnt = 1
    cntMatrix = frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret or cnt == max_frame:
            break

        cnt += 1
        cntMatrix = (cntMatrix * (cnt - 1) + frame) / cnt
        print(cnt)

    cap.release()
    bgr = cntMatrix.astype(np.uint8)

    return bgr

def top_gray_bg(file_name = "cctv.mp4", max_frame = None, check_cnt = 150):
    if max_frame is None:
        max_frame = -1
    
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    ret, frame = cap.read()

    h = frame.shape[0]
    w = frame.shape[1]

    cnt = 1

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr = grayFrame.copy()
    frames_saver = np.array([bgr])
    all_video = np.array([bgr])

    while True:
        ret, frame = cap.read()
        if not ret or cnt == max_frame:
            break

        cnt += 1
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_saver = np.append(frames_saver, np.array([grayFrame]), axis = 0)

        if cnt % check_cnt == 0:
            for i in range(h):
                for j in range(w):
                    bgr[i, j] = np.bincount(frames_saver[:, i, j]).argmax()
            if cnt == check_cnt:
                all_video = np.array([bgr])
            else:
                all_video = np.append(all_video, np.array([bgr]), axis = 0)
            frames_saver = np.array([bgr])
        print(cnt)

    cap.release()
    for i in range(h):
        for j in range(w):
            bgr[i, j] = np.bincount(all_video[:, i, j]).argmax()

    return bgr


bg = mean_bg("cctv1.mp4", None)


cv2.imshow('bg', bg)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite('bgr1.jpg', bg)
