###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
import cv2 as cv
import pandas as pd

# from argparse import RawTextHelpFormatter
import sys
import numpy as np
import utils
import _init_paths
from pathlib import Path
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from metrics_utils import *


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == "xywh":
        return BBFormat.XYWH
    elif argFormat == "xyrb":
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            "argument %s: invalid value. It must be either 'xywh' or 'xyrb'" % argName
        )


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append("argument %s: required argument" % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = "argument %s: required argument if %s is relative" % (
        argName,
        argInformed,
    )
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace("(", "").replace(")", "")
        args = arg.split(",")
        if len(args) != 2:
            errors.append(
                "%s. It must be in the format 'width,height' (e.g. '600,400')"
                % errorMsg
            )
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    "%s. It must be in INdiaTEGER the format 'width,height' (e.g. '600,400')"
                    % errorMsg
                )
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == "abs":
        return CoordinatesType.Absolute
    elif arg == "rel":
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append(
        "argument %s: invalid value. It must be either 'rel' or 'abs'" % argName
    )


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append("argument %s: invalid directory" % nameArg)
    elif (
        os.path.isdir(arg) is False
        and os.path.isdir(os.path.join(currentPath, arg)) is False
    ):
        errors.append("argument %s: directory does not exist '%s'" % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2

    # Define image size
    width = 448
    height = 448
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height, width, 3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        # cv2.imshow(key, image)
        # cv2.waitKey()
        cv2.imwrite("../results/pascal_rep_imgs/" + key, image)


def getBoundingBoxes(
    directory,
    isGT,
    bbFormat,
    coordType,
    allBoundingBoxes=None,
    allClasses=None,
    imgSize=(0, 0),
):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(" ", "") == "":
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = splitLine[0]  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat,
                )
            else:
                # idClass = int(splitLine[0]) #class
                idClass = splitLine[0]  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat,
                )
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = "0.1 (beta)"

parser = argparse.ArgumentParser(
    prog="Object Detection Metrics - Pascal VOC",
    description="This project applies the most popular metrics used to evaluate object detection "
    "algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, "
    "please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics",
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)",
)
# formatter_class=RawTextHelpFormatter)
parser.add_argument("-v", "--version", action="version", version="%(prog)s " + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
    "-gt",
    "--gtfolder",
    dest="gtFolder",
    default="../bounding_boxes/test/groundtruths",  # os.path.join(currentPath, 'groundtruths'),
    metavar="",
    help="folder containing your ground truth bounding boxes",
)
parser.add_argument(
    "-det",
    "--detfolder",
    dest="detFolder",
    default="../bounding_boxes/test/detections",  # os.path.join(currentPath, 'detections'),
    metavar="",
    help="folder containing your detected bounding boxes",
)
# Optional
parser.add_argument(
    "-t",
    "--threshold",
    dest="iouThreshold",
    type=float,
    default=0.5,
    metavar="",
    help="IOU threshold. Default 0.5",
)
parser.add_argument(
    "-gtformat",
    dest="gtFormat",
    metavar="",
    default="xywh",
    help="format of the coordinates of the ground truth bounding boxes: "
    "('xywh': <left> <top> <width> <height>)"
    " or ('xyrb': <left> <top> <right> <bottom>)",
)
parser.add_argument(
    "-detformat",
    dest="detFormat",
    metavar="",
    default="xywh",
    help="format of the coordinates of the detected bounding boxes "
    "('xywh': <left> <top> <width> <height>) "
    "or ('xyrb': <left> <top> <right> <bottom>)",
)
parser.add_argument(
    "-gtcoords",
    dest="gtCoordinates",
    default="abs",
    metavar="",
    help="reference of the ground truth bounding box coordinates: absolute "
    "values ('abs') or relative to its image size ('rel')",
)
parser.add_argument(
    "-detcoords",
    default="abs",
    dest="detCoordinates",
    metavar="",
    help="reference of the ground truth bounding box coordinates: "
    "absolute values ('abs') or relative to its image size ('rel')",
)
parser.add_argument(
    "-imgsize",
    dest="imgSize",
    metavar="",
    help="image size. Required if -gtcoords or -detcoords are 'rel'",
)
parser.add_argument(
    "-sp",
    "--savepath",
    default="../results/pascalvoc",
    dest="savePath",
    metavar="",
    help="folder where the plots are saved",
)
parser.add_argument(
    "-np",
    "--noplot",
    dest="showPlot",
    action="store_false",
    help="no plot is shown during execution",
)
parser.add_argument(
    "-dt",
    "--data_directory",
    dest="data_directory",
    default="../../../field_data/",
    help="path to images directory",
)
parser.add_argument(
    "--csv_path",
    default="../../../field_test0.csv",
    help="path to csv file with test data",
)
parser.add_argument(
    "--results_directory",
    default="../../../results/test_predictions_with_gt/",
    help="path to results directory",
)

parser.add_argument(
    "--conf",
    dest="confThreshold",
    type=float,
    default=0.5,
    metavar="",
    help="confidence threshold. Default 0.5",
)

args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, "-gtformat", errors)
detFormat = ValidateFormats(args.detFormat, "-detformat", errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, "-gt/--gtfolder", errors):
    gtFolder = ValidatePaths(args.gtFolder, "-gt/--gtfolder", errors)
else:
    # errors.pop()
    gtFolder = os.path.join(currentPath, "groundtruths")
    if os.path.isdir(gtFolder) is False:
        errors.append("folder %s not found" % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, "-gtCoordinates", errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, "-detCoordinates", errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, "-imgsize", "-gtCoordinates", errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, "-imgsize", "-detCoordinates", errors)
# Detection folder
if ValidateMandatoryArgs(args.detFolder, "-det/--detfolder", errors):
    detFolder = ValidatePaths(args.detFolder, "-det/--detfolder", errors)
else:
    # errors.pop()
    detFolder = os.path.join(currentPath, "detections")
    if os.path.isdir(detFolder) is False:
        errors.append("folder %s not found" % detFolder)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, "-sp/--savepath", errors)
else:
    savePath = os.path.join(currentPath, "results")
# Validate savePath
# If error, show error messages
if len(errors) != 0:
    print(
        """usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]"""
    )
    print("Object Detection Metrics: error(s): ")
    [print(e) for e in errors]
    sys.exit()

# Check if path to save results already exists and is not empty
if os.path.isdir(savePath) and os.listdir(savePath):
    key_pressed = ""
    while key_pressed.upper() not in ["Y", "N"]:
        print(f"Folder {savePath} already exists and may contain important results.\n")
        print(
            f"Enter 'Y' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!"
        )
        print(f"Or enter 'N' to abort and choose another folder to save the results.")
        key_pressed = input("")

    if key_pressed.upper() == "N":
        print("Process canceled")
        sys.exit()

# Clear folder and save results
shutil.rmtree(savePath, ignore_errors=True)
os.makedirs(savePath)
# Show plot during execution
showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFolder = %s' % gtFolder)
# print('detFolder = %s' % detFolder)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize
)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    detFolder,
    False,
    detFormat,
    detCoordType,
    allBoundingBoxes,
    allClasses,
    imgSize=imgSize,
)
allClasses.sort()

dr = args.data_directory
img_index = []
img_names = []
class_list = []
TP_list = []
FP_list = []
GT_list = []
FN_list = []
loc_FP_list = []
cls_FP_list = []
dbl_FP_list = []
prec_list = []
recall_list = []
FP_rate_list = []
count = 0
count_flag = 0

# clear results folder
shutil.rmtree(args.results_directory, ignore_errors=True)
os.makedirs(args.results_directory, exist_ok=True)


# create list of image files in the test set
test_images = pd.read_csv(args.csv_path)
test_images = list(test_images["File path"].unique())
test_images = [os.path.basename(t) for t in test_images]
# draw bounding boxes of gt and detections on data images. and create csv file with eval. metrics
for root, dirs, files in os.walk(dr):
    for index, file in enumerate(files):
        path = os.path.join(root, file)
        if file in test_images:
            img_name = Path(path)
            img_name = img_name.stem
            im = cv.imread(path)
            im = cv.resize(im, (896, 896))
            # print(img_name)
            # Add bounding boxes
            im = allBoundingBoxes.drawAllBoundingBoxes(im, img_name, args.confThreshold)
            cv.imwrite(args.results_directory + img_name + ".jpg", im)

            detections = allBoundingBoxes.calculateMetricsPerImage(
                img_name, iouThreshold / 2, args.confThreshold, MethodAveragePrecision
            )
            # print(f"image: {img_name} TP: {TP}, FP: {FP}, prec: {prec}, class: {cl}")
            if detections["cls exist"]:
                img_index.append(count)
                img_names.append(img_name)
                # class_list.append(c)
                TP_list.append(detections["trs TP"])
                FP_list.append(detections["trs FP"])
                loc_FP_list.append(detections["loc FP"])
                cls_FP_list.append(detections["cls FP"])
                dbl_FP_list.append(detections["dbl FP"])
                GT_list.append(detections["total GT"])
                FN_list.append(detections["total GT"] - detections["trs TP"])
                prec_list.append(detections["precision"])
                recall_list.append(detections["recall"])
                FP_rate_list.append(detections["FP rate"])
                count += 1

img_index = np.array(img_index)
img_names = np.array(img_names)
# class_list = np.array(class_list)
TP_arr = np.array(TP_list)
FP_arr = np.array(FP_list)
FP_rate_arr = np.array(FP_rate_list)
loc_FP_arr = np.array(loc_FP_list)
cls_FP_arr = np.array(cls_FP_list)
dbl_FP_arr = np.array(dbl_FP_list)
GT_arr = np.array(GT_list)
FN_arr = np.array(FN_list)
prec_arr = np.array(prec_list)
recall_arr = np.array(recall_list)
prec_avg = float(format(np.nanmean(prec_arr), ".2f"))
rec_avg = float(format((np.nanmean(recall_arr)), ".2f"))
data = np.column_stack(
    (
        img_index,
        img_names,
        GT_arr,
        TP_arr,
        FP_arr,
        FP_rate_arr,
        loc_FP_arr,
        cls_FP_arr,
        dbl_FP_arr,
        FN_arr,
        prec_arr,
        recall_arr,
    )
)
dataset = pd.DataFrame(
    {
        "Index": data[:, 0],
        "File path": data[:, 1],
        "GT": data[:, 2],
        "TP": data[:, 3],
        "FP": data[:, 4],
        "FP_rate": data[:, 5],
        "loc_FP": data[:, 6],
        "cls_FP": data[:, 7],
        "dbl_FP": data[:, 8],
        "FN": data[:, 9],
        "Precision": data[:, 10],
        "Recall": data[:, 11],
        "Precision avg": prec_avg,
        "Recall avg": rec_avg,
    }
)
dataset.to_csv("../../../eval_metrics.csv", index=False)

# ap_selected_classes = np.mean(prec_arr)
# ar_selected_classes = np.mean(recall_arr)
# print(f"ap for selected classes: {ap_selected_classes}, average recall for selected classes: {ar_selected_classes}")

# compute object/background accuracy:
total_tp = np.sum(TP_arr)
total_fp = np.sum(FP_arr)
total_loc_fp = np.sum(loc_FP_arr)
obj_error = np.divide(total_loc_fp, (total_tp + total_fp))
obj_accuracy = 1 - obj_error

# compute classification accuracy:
total_cls_fp = np.sum(cls_FP_arr)
cls_error = np.divide(total_cls_fp, (total_tp + total_fp))
cls_accuracy = 1 - cls_error

evaluator = Evaluator()
acc_AP = 0
validClasses = 0

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
    allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iouThreshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    savePath=savePath,
    showGraphic=False,
)

f = open(os.path.join(savePath, "results.txt"), "w")
f.write("Object Detection Metrics\n")
f.write("https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n")
f.write("Average Precision (AP), Precision and Recall per class:")

# each detection is a class
for metricsPerClass in detections:

    # Get metric values per each class
    cl = metricsPerClass["class"]
    ap = metricsPerClass["AP"]
    precision = metricsPerClass["precision"]
    recall = metricsPerClass["recall"]
    totalPositives = metricsPerClass["total positives"]
    total_TP = metricsPerClass["total TP"]
    total_FP = metricsPerClass["total FP"]

    if totalPositives > 0:
        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        prec = ["%.2f" % p for p in precision]
        rec = ["%.2f" % r for r in recall]
        ap_str = "{0:.2f}%".format(ap * 100)
        # ap_str = "{0:.4f}%".format(ap * 100)
        print("AP: %s (%s)" % (ap_str, cl))
        f.write("\n\nClass: %s" % cl)
        f.write("\nAP: %s" % ap_str)
        f.write("\nPrecision: %s" % prec)
        f.write("\nRecall: %s" % rec)

mAP = acc_AP / validClasses
mAP_str = "{0:.2f}%".format(mAP * 100)
print("mAP: %s" % mAP_str)
f.write("\n\n\nmAP: %s" % mAP_str)
print(
    f"obj/bg accuracy: {obj_accuracy:.3f}, classifcation accuracy: {cls_accuracy:.3f}"
)
