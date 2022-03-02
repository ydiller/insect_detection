import cv2 as cv
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import utils
from pathlib import Path


def transformations(path, opt):
    src = cv.imread(path)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.bilateralFilter(src_gray, 5, 75, 75)
    width = int(src_gray.shape[1] * opt.scale_percent / 100)
    height = int(src_gray.shape[0] * opt.scale_percent / 100)
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


def is_good_shape(bound_rect, opt):
    # Determine whether rectangle is in a good shape or not
    if opt.lower_boundary <= get_area(bound_rect) <= opt.upper_boundary:
        if bound_rect[2] < opt.upper_width and bound_rect[3] < opt.upper_width:
            return True
    return False


def kill_overlapping_boxes(bound_rectangles, threshold):
    for i, rect1 in enumerate(bound_rectangles):
        for j, rect2 in enumerate(bound_rectangles):
            # print(intersect_over_union(rect1, rect2))
            if intersect_over_union(rect1, rect2) > threshold and i != j:
                # print('entered at ', intersect_over_union(rect1, rect2))
                if get_area(rect1) > get_area(rect2):
                    del bound_rectangles[j]
                else:
                    del bound_rectangles[i]

    return bound_rectangles


def thresh_callback(src_gray, opt, threshold):
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    _, contours, _ = cv.findContours(
        canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # con = cv.drawContours(src_gray, contours, -1, (0, 0, 255),3)
    bound_rectangles = []
    scale = 100 / opt.scale_percent
    for c in contours:
        bound_rect = cv.boundingRect(cv.approxPolyDP(c, 3, True))
        if is_good_shape(bound_rect, opt):
            bound_rectangles.append(
                (
                    int(bound_rect[0] * scale),
                    int(bound_rect[1] * scale),
                    int(bound_rect[2] * scale),
                    int(bound_rect[3] * scale),
                )
            )
    kill_overlapping_boxes(bound_rectangles, 0.4)
    return bound_rectangles


def drawbox(csv_path, results_folder):
    df = pd.read_csv(csv_path)
    # names=df['File path'].unique().tolist()
    names = df["File path"].unique()
    data_frame_dict = {elem: pd.DataFrame for elem in names}
    # df = df[1:84]
    for key in data_frame_dict.keys():
        data_frame_dict[key] = df[:][df["File path"] == key]
        path = data_frame_dict[key].iloc[0][1]
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for index, row in data_frame_dict[key].iterrows():
            x = row["X"]
            y = row["Y"]
            w = row["W"]
            h = row["H"]
            img_with_rects = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        plt.imsave(
            results_folder + path[-15:-4] + "-high_quality.jpg",
            img_with_rects,
            cmap="gray",
        )


# Main function for finding bounding boxes of flies in the dataset, using canny algorithm.
# The bounding boxes are written to a csv file (X,Y,W,H) including the image path.
# The 'drawbox' function draws the bounding boxes over the dataset images.
# on command line use: --scale percent 10  --upper_boundary 600  --lower_boundary 200
# --upper_width 70 --results '../results/'  --data_directory '../data/  --csv_path '../bounding_boxes.csv'
def main():
    opt = utils.parse_flags()
    dr = opt.data_directory
    all_rectangles = []
    paths = []
    labels = []
    image_index = (
        []
    )  # index for all the images, all the boxes belongs to certain index have the same index.
    count = 0
    # ratio between original size to model input size:
    x_ratio = 3280 / 896
    y_ratio = 2464 / 896
    for root, dirs, files in os.walk(dr):
        for index, file in enumerate(files):
            if file != "desktop.ini":  # get over windows problem
                path = os.path.join(root, file)
                img_name = Path(path)
                img_name = img_name.stem
                src_gray = transformations(path, opt)
                label = 1 if file[7:9] == "cc" else 2
                rectangles = thresh_callback(src_gray, opt, threshold=80)
                # create txt file with bbox data. The file is used for evaluation metrics.
                bbox_file = open(
                    opt.txt_path + "groundtruths/" + img_name + ".txt", "w"
                )
                for rect in rectangles:
                    paths.append(path)
                    labels.append(label)
                    image_index.append(count)
                    x, y, w, h = rect
                    line = [
                        f"{label} {x/x_ratio} {y/y_ratio} {w/x_ratio} {h/y_ratio}\n"
                    ]
                    bbox_file.writelines(line)  # add new bbox to txt file

                all_rectangles.append(rectangles)
                count += 1

    all_rectangles = list(itertools.chain.from_iterable(all_rectangles))
    rect_arr = np.array(all_rectangles)
    paths_arr = np.array(paths)
    labels_arr = np.array(labels)
    index_arr = np.array(image_index)
    data = np.column_stack((index_arr, paths_arr, rect_arr, labels_arr))
    dataset = pd.DataFrame(
        {
            "Index": data[:, 0],
            "File path": data[:, 1],
            "X": data[:, 2],
            "Y": data[:, 3],
            "W": data[:, 4],
            "H": data[:, 5],
            "Label": data[:, 6],
        }
    )
    dataset.to_csv(opt.csv_path, index=False)
    # drawbox(opt.csv_path, opt.results) # draw and save the images included in the csv file with the bboxes

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
