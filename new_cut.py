import cv2
import numpy as np
import copy
import math

def get_to_edge(start, direction, threshold, h_frame, w_frame):
    x = start[0]
    y = start[1]

    while True:
        black_cnt = (w_frame[1] - w_frame[0]) * (h_frame[1] - h_frame[0]) - np.count_nonzero(threshold[
        x + h_frame[0]: x + h_frame[1], 
        y + w_frame[0]: y + w_frame[1]])

        if black_cnt >= 3:
            break

        x += direction[0]
        y += direction[1]

    return (x, y)

def get_bound(threshold):
    p1 = get_to_edge((0, threshold.shape[1] // 2), (1, 0), threshold, 
                     (0, 20), (-5, 5))
    p2 = get_to_edge((threshold.shape[0] - 1, threshold.shape[1] // 2), (-1, 0), threshold,
                     (-20, 0), (-5, 5))

    p3 = get_to_edge((threshold.shape[0] // 2, 0), (0, 1), threshold, 
                     (-5, 5), (0, 20))
    p4 = get_to_edge((threshold.shape[0] // 2, threshold.shape[1] - 1), (0, -1), threshold,
                     (-5, 5), (-20, 0))

    p5 = get_to_edge((0, 0), (3, 4), threshold, (0, 20), (0, 20))
    
    p6 = get_to_edge((threshold.shape[0] - 1, threshold.shape[1] - 1), (-3, -4), threshold,
                     (-10, 0), (-10, 0))

    # cv2.circle(img, p1[::-1], 10, (0, 0, 255), -1)
    # cv2.circle(img, p2[::-1], 10, (0, 0, 255), -1)
    # cv2.circle(img, p3[::-1], 10, (0, 0, 255), -1)
    # cv2.circle(img, p4[::-1], 10, (0, 0, 255), -1)
    # cv2.circle(img, p5[::-1], 10, (0, 0, 255), -1)
    # cv2.circle(img, p6[::-1], 10, (0, 0, 255), -1)

    return p1[0], p2[0], p3[1], p4[1], cv2.fitEllipse(np.array([
        p1[::-1], 
        p2[::-1], 
        p3[::-1], 
        p4[::-1], 
        p5[::-1],
        p6[::-1]]))

def to_int_tup(tup):
    return (int(tup[0]), int(tup[1]))

def get_axes_length(tup):
    return (int(tup[0] / 2), int(tup[1] / 2))

def pre_process(path):
    img = cv2.imread(path)
    gray = cv2.imread(path, 0)

    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("thres.png", thresholded)

    h_start, h_end, w_start, w_end, _ = get_bound(thresholded)

    #cv2.imwrite(r'points.png', img)

    img = img[h_start:h_end, w_start:w_end]

    rows, cols = (h_end - h_start), (w_end - w_start)

    win_size = 200
    norm_size = 300

    if rows > cols:
        cols = int(cols * (win_size / rows))
        rows = win_size
    else:
        rows = int(rows * (win_size / cols))
        cols = win_size

    img = cv2.resize(img, (cols, rows))

    cols_pad_x, cols_pad_y = math.ceil((norm_size-cols)/2.0), int(math.floor((norm_size-cols)/2.0))
    rows_pad_x, rows_pad_y = int(math.ceil((norm_size-rows)/2.0)), int(math.floor((norm_size-rows)/2.0))

    return img[h_start + cols_pad_x: h_end + cols_pad_y, w_start + rows_pad_x: w_end]

import os

dir = r"D:\Computer vision\Images\Rupee\DataSet"

count = 0
for dirpath, dnames, fnames in os.walk(dir):
    for f in fnames:
        full_path = os.path.join(dirpath, f)
        
        processed = pre_process(full_path)
        cv2.imwrite(fr"D:\Computer vision\Images\Rupee\Processed2\{f}.png", processed)