import json
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from pathlib import Path


def drawbox(csv_path, results_folder):
    df = pd.read_csv(csv_path)
    names = df['File path'].unique()
    data_frame_dict = {elem: pd.DataFrame for elem in names}
    category_id_to_name = {1: 'PEACH-FF', 2: 'S-HOUSE-FLY', 3: 'L-HOUSE-FLY', 4: 'OTHER', 5: 'MEDFLY',
                           6: 'SPIDER', 7: 'L-ANT', 8: 'Ants', 9: 'Bee', 10: 'LACEWING '}
    category_id_to_color = {1: (255, 0, 0), 2: (255, 128, 0), 3: (255, 255, 0), 4: (128, 255, 0), 5: (0, 255, 0),
                           6: (0, 255, 128), 7: (0, 255, 255), 8: (0, 128, 255), 9: (0, 0, 255), 10: (127, 0, 255)}
    for key in data_frame_dict.keys():
        data_frame_dict[key] = df[:][df['File path'] == key]
        path = data_frame_dict[key].iloc[0][1]
        img_name = Path(path)
        img_name = img_name.stem
        img = cv.imread(path)
        for index, row in data_frame_dict[key].iterrows():
            x = row['X']
            y = row['Y']
            w = row['W']
            h = row['H']
            label = int(row['Label'])
            class_name = category_id_to_name[label]
            color = category_id_to_color[label]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 4)
            cv.putText(img, class_name, (x, y), cv.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255),
                       thickness=3)  # Write the prediction class
        # img = cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)
        # plt.imsave(results_folder + path[-15:-4] + '-high_quality.jpg', img, cmap='gray')
        plt.imsave(results_folder + img_name + '_gt.jpg', img)

def main():
    opt = utils.parse_flags()
    dr = opt.json_directory
    paths = []
    labels = []
    image_index = []  # index for all the images, all the boxes belongs to certain index have the same index.
    x_list = []
    y_list = []
    w_list = []
    h_list = []
    count = 0
    for root, dirs, files in os.walk(dr):
        for index, file in enumerate(files):
            if file != 'desktop.ini':  # get over windows problem
                path = os.path.join(root, file)
                with open(path, "r") as read_file:
                    json_file = json.load(read_file)
                image_name = list(json_file.keys())[0]
                json_data = json_file[image_name]
                # create txt file with bbox data. The file is used for evaluation metrics.
                bbox_file = open(opt.txt_path + "groundtruths/" + image_name + ".txt", "w")
                for bbox in json_data[:-1]:
                    box_data = bbox["points"]
                    x = int(box_data["x1"])
                    y = int(box_data["y1"])
                    w = int(box_data["x2"]) - int(box_data["x1"])
                    h = int(box_data["y2"]) - int(box_data["y1"])
                    label = int(bbox["classId"])
                    # print(f"box data: {box_data}, label: {label}")
                    if 0 < label < 12:
                        image_index.append(count)
                        paths.append(opt.data_directory+image_name)
                        labels.append(label)
                        x_list.append(x)
                        y_list.append(y)
                        w_list.append(w)
                        h_list.append(h)
                        line = [f"{label} {x} {y} {h} {w}\n"]
                        bbox_file.writelines(line)  # add new bbox to txt file
                count += 1
        paths_arr = np.array(paths)
        labels_arr = np.array(labels)
        index_arr = np.array(image_index)
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        w_arr = np.array(w_list)
        h_arr = np.array(h_list)
        data = np.column_stack((index_arr, paths_arr, x_arr, y_arr, w_arr, h_arr, labels_arr))
        dataset = pd.DataFrame(
            {'Index': data[:, 0], 'File path': data[:, 1], 'X': data[:, 2], 'Y': data[:, 3], 'W': data[:, 4],
             'H': data[:, 5], 'Label': data[:, 6]})
        dataset.to_csv(opt.csv_path, index=False)
        drawbox(opt.csv_path, opt.results_directory) # draw and save the images included in the csv file with the bboxes

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

