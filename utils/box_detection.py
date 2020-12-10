import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    back = _compute_background(cap)
    corners = _detect_rectangle(back, save_path)
    return corners

def _compute_background(cap):
    print("Computing background")
    # data = np.zeros(shape=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(4)), int(cap.get(3)), 3))
    data = np.zeros(shape=(100, int(cap.get(4)), int(cap.get(3)), 3))
    idxs = np.ceil(np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT), 100))

    i = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[i])
        ret, frame = cap.read()
        if ret:
            data[i] = np.squeeze(frame)
        else:
            break
        i += 1

    med = np.median(data, axis=0)
    # kernel = np.ones((5, 5), np.uint8)
    # back = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel)
    back = med
    # plt.imshow(back/255, cmap='gray')
    # plt.show()
    print("Background computed")
    return np.float32(back)

def _detect_rectangle(img, save_path):
    print("Detecting corners")
    edges = cv2.Canny(img.astype(np.uint8), 80, 250)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    # plt.imshow(edges / 255, cmap='gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 130000 and area < 170000:
            im = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
            # plt.imshow(im / 255, cmap='gray')
            # plt.show()
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            im = cv2.drawContours(img.copy(), [approx], 0, (0, 0, 255), 2)
            plt.imshow(im / 255, cmap='gray')
            plt.savefig(save_path)
            # plt.show()

            if approx.shape[0] != 4:
                raise Exception("Shape detected with more than 4 corners!")
            print("Corners detected")

            return np.squeeze(approx)

    raise Exception("No field found in the video.")

