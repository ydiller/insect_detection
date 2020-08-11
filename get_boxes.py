import cv2 as cv
import os
import numpy as np
import pickle
from train_val_splitting import split_into_train_val

SCALE_PERCENT = 10
UPPER_AREA_BOUNDARY = SCALE_PERCENT * 60
LOWER_AREA_BOUNDARY = SCALE_PERCENT * 20


def transformations(src):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    width = int(src_gray.shape[1] * SCALE_PERCENT / 100)
    height = int(src_gray.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)

    src_gray = cv.resize(src_gray, dim, interpolation=cv.INTER_AREA)
    return src_gray


def get_area(rect):
    return rect[2] * rect[3]


def intersect_over_union(bound_rect1, bound_rect2):
    x1, y1, w1, h1 = bound_rect1
    x2, y2, w2, h2 = bound_rect2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    i = w_intersection * h_intersection
    u = w1 * h1 + w2 * h2 - i  # Union = Total Area - I
    return i / u


def is_good_shape(bound_rect):
    # Determine whether rectangle is in a good shape or not
    if UPPER_AREA_BOUNDARY >= get_area(bound_rect) >= LOWER_AREA_BOUNDARY:
        if bound_rect[2] < 7 * SCALE_PERCENT and bound_rect[3] < 7 * SCALE_PERCENT:
            return True
    return False


def kill_overlapping_boxes(bound_rectangles):
    for i in bound_rectangles:
        for j in bound_rectangles:
            if intersect_over_union(i, j) > 0.4 and i != j:
                if get_area(i) > get_area(j):
                    bound_rectangles.remove(j)
                else:
                    bound_rectangles.remove(i)


def thresh_callback(threshold, src_gray, src, name):  # , splits was here
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    bound_rectangles = []
    original_rectangles = []

    for c in contours:
        bound_rect = cv.boundingRect(cv.approxPolyDP(c, 3, True))
        if is_good_shape(bound_rect):
            bound_rectangles.append(bound_rect)

    kill_overlapping_boxes(bound_rectangles)

    for rect in bound_rectangles:
        # Append original rectangles, resized and converted to [xmin, ymin, xmax, ymax]
        original_rectangles.append((rect[0] * SCALE_PERCENT, rect[1] * SCALE_PERCENT,
                                    (rect[0] + rect[2]) * SCALE_PERCENT, (rect[1] + rect[3]) * SCALE_PERCENT))

    # boxes = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    #
    # for bound_rect in bound_rectangles:
    #     color = 0
    #     cv.rectangle(src_gray, (bound_rect[0], bound_rect[1]),
    #                  ((bound_rect[0] + bound_rect[2]), (bound_rect[1] + bound_rect[3])), color, 1)
    #
    # cv.imshow(f'{name}', src_gray)
    #
    # color = 1
    # for rectangle in original_rectangles:
    #     cv.rectangle(boxes, (rectangle[0], rectangle[1]),
    #                  ((rectangle[0] + rectangle[2]), (rectangle[1] + rectangle[3])), color, -1)
    #     color += 1
    #
    #

    pickle_name = name + '.p'
    pickle_path = os.path.join('/home/itay/PycharmProjects/flies/boxes', pickle_name)
    pickle.dump(original_rectangles, open(pickle_path, 'wb'))
    # cv.imshow(f'{name} boxes', boxes)


def main():
    # splits = split_into_train_val("./data")
    dr = "./images"
    for root, dirs, files in os.walk(dr):
        for index, file in enumerate(files):
            path = os.path.join(root, file)
            src = cv.imread(path)
            src_gray = transformations(src)
            threshold = 90
            name = file.split('.')[0]  # without jpg
            thresh_callback(threshold, src_gray, src, name)  # , splits was here
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
