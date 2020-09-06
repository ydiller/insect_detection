import cv2 as cv
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

SCALE_PERCENT = 10
UPPER_AREA_BOUNDARY = SCALE_PERCENT * 60
LOWER_AREA_BOUNDARY = SCALE_PERCENT * 20


def transformations(path):
    src = cv.imread(path)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.bilateralFilter(src_gray, 5, 75, 75)
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
    if LOWER_AREA_BOUNDARY <= get_area(bound_rect) <= UPPER_AREA_BOUNDARY:
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


def thresh_callback(src_gray, threshold):
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # con = cv.drawContours(src_gray, contours, -1, (0, 0, 255),3)
    bound_rectangles = []
    scale = 100 / SCALE_PERCENT
    for c in contours:
        bound_rect = cv.boundingRect(cv.approxPolyDP(c, 3, True))
        if is_good_shape(bound_rect):
            bound_rectangles.append((int(bound_rect[0] * scale), int(bound_rect[1] * scale),
                                     int(bound_rect[2] * scale), int(bound_rect[3] * scale)))
    kill_overlapping_boxes(bound_rectangles)
    return bound_rectangles

def drawbox(csv_path,results_folder):
    df = pd.read_csv(csv_path)
    # names=df['File path'].unique().tolist()
    names = df['File path'].unique()
    data_frame_dict = {elem: pd.DataFrame for elem in names}
    # df = df[1:84]
    for key in data_frame_dict.keys():
        data_frame_dict[key] = df[:][df['File path'] == key]
        path = data_frame_dict[key].iloc[0][0]
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for index, row in data_frame_dict[key].iterrows():
            x = row['X']
            y = row['Y']
            w = row['W']
            h = row['H']
            img_with_rects = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        plt.imsave(results_folder + path[-15:-4] + '-high_quality.jpg', img_with_rects, cmap='gray')

def main():
    dr = "../bad_data/"
    all_rectangles = []
    paths = []
    labels = []
    image_index = []  # index for all the images, all the boxes belongs to certain index have the same index.
    count = 0
    for root, dirs, files in os.walk(dr):
        for index, file in enumerate(files):
            if file != 'desktop.ini':  # get over windows problem
                path = os.path.join(root, file)
                src_gray = transformations(path)
                label = 1 if file[7:9] == 'cc' else 2
                rectangles = thresh_callback(src_gray, 80)
                for rect in rectangles:
                    paths.append(path)
                    labels.append(label)
                    image_index.append(count)
                all_rectangles.append(rectangles)
                count += 1

    all_rectangles = list(itertools.chain.from_iterable(all_rectangles))
    rect_arr = np.array(all_rectangles)
    paths_arr = np.array(paths)
    labels_arr = np.array(labels)
    index_arr = np.array(image_index)
    data = np.column_stack((index_arr, paths_arr, rect_arr, labels_arr))
    dataset = pd.DataFrame(
        {'Index': data[:, 0], 'File path': data[:, 1], 'X': data[:, 2], 'Y': data[:, 3], 'W': data[:, 4],
         'H': data[:, 5], 'Label': data[:, 6]})
    csv_path = '../bounding_boxes.csv'
    dataset.to_csv(csv_path, index=False)
    drawbox(csv_path, '../results/')

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
