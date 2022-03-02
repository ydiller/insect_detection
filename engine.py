import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
from statistics import mean
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from datetime import datetime
from collections import Counter
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    running_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0

    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        # original equal loss
        losses = sum(loss for loss in loss_dict.values())

        # weighted loss
        # losses = loss_dict['loss_classifier'] + loss_dict['loss_objectness']
        # + 0.5*loss_dict['loss_box_reg'] + 0.5*loss_dict['loss_rpn_box_reg']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # previous version: losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        losses_reduced = (
            loss_dict_reduced["loss_classifier"].item()
            + loss_dict_reduced["loss_objectness"].item()
            + 0.5 * loss_dict_reduced["loss_box_reg"].item()
            + 0.5 * loss_dict_reduced["loss_rpn_box_reg"].item()
        )
        loss_value = (
            losses_reduced  # previous version: loss_value = losses_reduced.item()
        )
        running_loss += loss_value

        loss_classifier += loss_dict_reduced["loss_classifier"].item()
        loss_box_reg += loss_dict_reduced["loss_box_reg"].item()
        loss_objectness += loss_dict_reduced["loss_objectness"].item()
        loss_rpn_box_reg += loss_dict_reduced["loss_rpn_box_reg"].item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    running_loss = running_loss / len(data_loader)
    loss_classifier = loss_classifier / len(data_loader)
    loss_box_reg = loss_box_reg / len(data_loader)
    loss_objectness = loss_objectness / len(data_loader)
    loss_rpn_box_reg = loss_rpn_box_reg / len(data_loader)
    return (
        loss_classifier,
        loss_box_reg,
        loss_objectness,
        loss_rpn_box_reg,
        running_loss,
    )


def get_val_loss(model, data_loader_val, device):
    model.train()
    running_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    for images, targets, _ in data_loader_val:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            val_loss_dict = model(images, targets)
            val_loss_dict_reduced = utils.reduce_dict(val_loss_dict)
            losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            running_loss += loss_value
            loss_classifier += val_loss_dict_reduced["loss_classifier"].item()
            loss_box_reg += val_loss_dict_reduced["loss_box_reg"].item()
            loss_objectness += val_loss_dict_reduced["loss_objectness"].item()
            loss_rpn_box_reg += val_loss_dict_reduced["loss_rpn_box_reg"].item()

    running_loss = running_loss / len(data_loader_val)
    loss_classifier = loss_classifier / len(data_loader_val)
    loss_box_reg = loss_box_reg / len(data_loader_val)
    loss_objectness = loss_objectness / len(data_loader_val)
    loss_rpn_box_reg = loss_rpn_box_reg / len(data_loader_val)
    return (
        loss_classifier,
        loss_box_reg,
        loss_objectness,
        loss_rpn_box_reg,
        running_loss,
    )


def get_scores(model, data_loader, device):
    model.eval()
    running_accuracy = 0
    for images, targets, _ in data_loader:
        images = [image.to(device) for image in images]

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        boxes = list(pred[0]["boxes"].detach().cpu().numpy())
        scores = list(pred[0]["scores"].detach().cpu().numpy())
        accuracy = sum(score for score in scores) / len(boxes)
        running_accuracy += accuracy

    running_accuracy = running_accuracy / len(data_loader)
    return running_accuracy


def get_distance(predicted_center, gt_center):
    gt_x, gt_y = gt_center
    pred_x, pred_y = predicted_center
    distance = (pred_y - gt_y) ** 2 + (pred_x - gt_x) ** 2
    return distance


def intersect_over_union(bound_rect1, bound_rect2):

    x1_1, y1_1, x1_2, y1_2 = bound_rect1
    x2_1, y2_1, x2_2, y2_2 = bound_rect2
    w1 = x1_2 - x1_1
    h1 = y1_2 - y1_1
    w2 = x2_2 - x2_1
    h2 = y2_2 - y2_1
    w_intersection = min(x1_1 + w1, x2_1 + w2) - max(x1_1, x2_1)
    h_intersection = min(y1_1 + h1, y2_1 + h2) - max(y1_1, y2_1)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    i = w_intersection * h_intersection
    u = w1 * h1 + w2 * h2 - i  # Union = Total Area - I
    return i / u


def get_accuracy(model, data_loader, device, score_threshold=0.7, iou_threshold=0.5):
    model.eval()
    accuracy = []
    for images, targets, _ in data_loader:
        images = [image.to(device) for image in images]
        # targets = [target.to(device) for target in targets]
        running_accuracy = 0

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        labels = list(pred[0]["labels"].cpu().numpy())
        boxes = list(pred[0]["boxes"].detach().cpu().numpy())
        scores = list(pred[0]["scores"].detach().cpu().numpy())

        gt_boxes = targets[0]["boxes"]
        gt_labels = targets[0]["labels"]
        gt_boxes_center = []
        pred_boxes_center = []
        # threshold_distance = (img_size / 100) ** 2 + (img_size / 100) ** 2
        threshold_distance = 5
        for i, gt_box in enumerate(gt_boxes):
            ioumax = 0
            gt_index = -1  # index of ground truth box
            pred_index = -2  # index of detected box
            for j, pred_box in enumerate(boxes):
                if scores[j] > score_threshold:
                    iou = intersect_over_union(gt_box, pred_box)
                    # print(f'iou {j}: {iou}')
                    if iou > iou_threshold:
                        if iou > ioumax:
                            ioumax = iou  # ioumax stores the maximum iou among the detected boxes
                            gt_index = (
                                i  # stores the index of the gt box with the maximal iou
                            )
                            pred_index = j  # stores the index of the detected box with the maximal iou
            if gt_index != -1 and pred_index != -2:
                if (gt_labels[gt_index] == labels[pred_index]) and (
                    ioumax > iou_threshold
                ):
                    running_accuracy += 1
        accuracy.append(running_accuracy / len(gt_boxes))
    total_accuracy = mean(accuracy)

    return total_accuracy


def write_detected_boxes(model, data_loader, device, opt, mode=""):
    model.eval()
    accuracy = []
    for images, targets, img_name in data_loader:
        images = [image.to(device) for image in images]
        # targets = [target.to(device) for target in targets]
        running_accuracy = 0

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        labels = list(pred[0]["labels"].cpu().numpy())
        boxes = list(pred[0]["boxes"].detach().cpu().numpy())
        scores = list(pred[0]["scores"].detach().cpu().numpy())
        bbox_file = open(
            opt.txt_path + mode + "detections/" + img_name[0] + ".txt", "w"
        )
        for i, pred_box in enumerate(boxes):
            label = labels[i]
            score = scores[i]
            x, y, x2, y2 = pred_box
            line = [f"{label} {score} {x} {y} {x2-x} {y2-y}\n"]
            bbox_file.writelines(line)


def write_field_detected_boxes(model, data_loader, device, opt, mode=""):
    model.eval()
    accuracy = []
    for images, targets, img_name in data_loader:
        images = [image.to(device) for image in images]
        # targets = [target.to(device) for target in targets]
        running_accuracy = 0

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        labels = list(pred[0]["labels"].cpu().numpy())
        boxes = list(pred[0]["boxes"].detach().cpu().numpy())
        scores = list(pred[0]["scores"].detach().cpu().numpy())
        bbox_file = open(
            opt.txt_path + mode + "detections/" + img_name[0] + ".txt", "w"
        )
        for i, pred_box in enumerate(boxes):
            # if(labels[i]==1):
            #     label = 5
            # elif (labels[i] == 2):
            #     label = 1
            label = labels[i]  # on full field train
            score = scores[i]
            x, y, x2, y2 = pred_box
            line = [f"{label} {score} {x} {y} {x2-x} {y2-y}\n"]
            bbox_file.writelines(line)


def write_test_field_detected_boxes(model, data_loader, device, opt, mode=""):
    """
    function to be used for real app where no labels are available
    """
    model.eval()
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    os.mkdir(opt.results_directory + current_time)
    # results_file = open(f"{opt.results_directory}{current_time}/{current_time}.txt", "w")
    conf_threshold = float(opt.results_thresh)
    final_label_list = []
    final_amount_list = []
    final_names_list = []
    for images, targets, img_name in data_loader:
        images = [image[0].to(device) for image in images]
        with torch.no_grad():
            # model = model.cuda()
            model = model.to(device)
            pred = model(images)
        labels = list(pred[0]["labels"].cpu().numpy())
        boxes = list(pred[0]["boxes"].detach().cpu().numpy())
        scores = list(pred[0]["scores"].detach().cpu().numpy())
        filtered_bbox = []
        filtered_labels = []
        filtered_scores = []
        [
            filtered_bbox.append(b)
            for i, b in enumerate(boxes)
            if scores[i] >= conf_threshold
        ]
        [
            filtered_labels.append(l)
            for i, l in enumerate(labels)
            if scores[i] >= conf_threshold
        ]
        [
            filtered_scores.append(s)
            for i, s in enumerate(scores)
            if scores[i] >= conf_threshold
        ]
        img_with_pred = drawbox_from_prediction(
            images[0].cpu().numpy(), filtered_bbox, filtered_scores, filtered_labels
        )
        plt.imsave(
            f"{opt.results_directory}{current_time}/{img_name[0]}.jpg", img_with_pred
        )
        labels_counter = Counter(filtered_labels)
        for item, value in zip(labels_counter.keys(), labels_counter.values()):
            final_label_list.append(item)
            final_amount_list.append(value)
            final_names_list.append(img_name[0])
    names_arr = np.array(final_names_list)
    labels_arr = np.array(final_label_list)
    amount_arr = np.array(final_amount_list)
    data = np.column_stack((names_arr, labels_arr, amount_arr))
    dataset = pd.DataFrame(
        {"File name": data[:, 0], "Label": data[:, 1], "Amount": data[:, 2]}
    )
    dataset.to_csv(f"{opt.results_directory}{current_time}/results.csv", index=False)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def drawbox_from_prediction(img, boxes, scores, labels):
    category_id_to_name = {
        1: "PEACH-FF",
        2: "S-HOUSE-FLY",
        3: "L-HOUSE-FLY",
        4: "OTHER",
        5: "MEDFLY",
        6: "SPIDER",
        7: "L-ANT",
        8: "Ants",
        9: "Bee",
        10: "LACEWING ",
        11: "ORIENTAL-FF",
    }
    # category_id_to_color = {1: (255, 0, 0), 2: (255, 128, 0), 3: (255, 255, 0), 4: (128, 255, 0), 5: (0, 255, 0),
    #                         6: (0, 255, 128), 7: (0, 255, 255), 8: (0, 128, 255), 9: (0, 0, 255), 10: (127, 0, 255),
    #                         11: (255, 0, 255)}
    # category_id_to_name = {1: 'MEDFLY', 2: 'PEACH-FF'}  # temporary dict for lab dataset
    # category_id_to_color = {1: (0, 255, 0), 2: (0, 0, 255)}
    img = (img * 255).astype(np.uint8)
    img = np.moveaxis(img, 0, -1)
    img = cv.UMat(img).get()
    for j in range(len(boxes)):
        x1 = int(boxes[j][0])
        y1 = int(boxes[j][1])
        x2 = int(boxes[j][2])
        y2 = int(boxes[j][3])
        class_name = category_id_to_name[labels[j]]
        if labels[j] == 1:
            color = (0, 153, 0)
        elif labels[j] == 5:
            color = (153, 0, 0)
        else:
            color = (0, 0, 153)
        score = str(format(scores[j], ".2f"))
        cv.rectangle(
            img, (x1, y1), (x2, y2), color=color, thickness=1
        )  # Draw Rectangle with the coordinates
        cv.putText(
            img,
            class_name,
            (x1, y1),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            thickness=1,
        )  # Write the prediction class
    return img
