import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
from statistics import mean
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
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
        losses_reduced = (loss_dict_reduced['loss_classifier'].item() + loss_dict_reduced['loss_objectness'].item()
        + 0.5*loss_dict_reduced['loss_box_reg'].item() + 0.5*loss_dict_reduced['loss_rpn_box_reg'].item())
        loss_value = losses_reduced # previous version: loss_value = losses_reduced.item()
        running_loss += loss_value

        loss_classifier += loss_dict_reduced['loss_classifier'].item()
        loss_box_reg += loss_dict_reduced['loss_box_reg'].item()
        loss_objectness += loss_dict_reduced['loss_objectness'].item()
        loss_rpn_box_reg += loss_dict_reduced['loss_rpn_box_reg'].item()

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

    running_loss = running_loss/len(data_loader)
    loss_classifier = loss_classifier/len(data_loader)
    loss_box_reg = loss_box_reg/len(data_loader)
    loss_objectness = loss_objectness/len(data_loader)
    loss_rpn_box_reg = loss_rpn_box_reg/len(data_loader)
    return loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg, running_loss


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
            loss_classifier += val_loss_dict_reduced['loss_classifier'].item()
            loss_box_reg += val_loss_dict_reduced['loss_box_reg'].item()
            loss_objectness += val_loss_dict_reduced['loss_objectness'].item()
            loss_rpn_box_reg += val_loss_dict_reduced['loss_rpn_box_reg'].item()

    running_loss = running_loss / len(data_loader_val)
    loss_classifier = loss_classifier/len(data_loader_val)
    loss_box_reg = loss_box_reg/len(data_loader_val)
    loss_objectness = loss_objectness/len(data_loader_val)
    loss_rpn_box_reg = loss_rpn_box_reg/len(data_loader_val)
    return loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg, running_loss


def get_scores(model, data_loader, device):
    model.eval()
    running_accuracy = 0
    for images, targets , _ in data_loader:
        images = [image.to(device) for image in images]

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        boxes = list(pred[0]['boxes'].detach().cpu().numpy())
        scores = list(pred[0]['scores'].detach().cpu().numpy())
        accuracy = sum(score for score in scores)/len(boxes)
        running_accuracy += accuracy

    running_accuracy = running_accuracy/len(data_loader)
    return running_accuracy


def get_distance(predicted_center, gt_center):
    gt_x, gt_y = gt_center
    pred_x, pred_y = predicted_center
    distance = (pred_y - gt_y)**2 + (pred_x - gt_x)**2
    return distance


def intersect_over_union(bound_rect1, bound_rect2):

    x1_1, y1_1, x1_2, y1_2 = bound_rect1
    x2_1, y2_1, x2_2, y2_2 = bound_rect2
    w1 = x1_2-x1_1
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

        labels = list(pred[0]['labels'].cpu().numpy())
        boxes = list(pred[0]['boxes'].detach().cpu().numpy())
        scores = list(pred[0]['scores'].detach().cpu().numpy())

        gt_boxes = targets[0]['boxes']
        gt_labels = targets[0]['labels']
        gt_boxes_center = []
        pred_boxes_center = []
        # threshold_distance = (img_size / 100) ** 2 + (img_size / 100) ** 2
        threshold_distance = 5
        for i, gt_box in enumerate(gt_boxes):
            ioumax = 0
            gt_index = -1 # index of ground truth box
            pred_index = -2 # index of detected box
            for j, pred_box in enumerate(boxes):
                if scores[j] > score_threshold:
                    iou = intersect_over_union(gt_box, pred_box)
                    #print(f'iou {j}: {iou}')
                    if iou > iou_threshold:
                        if iou > ioumax:
                            ioumax = iou  # ioumax stores the maximum iou among the detected boxes
                            gt_index = i  # stores the index of the gt box with the maximal iou
                            pred_index = j  # stores the index of the detected box with the maximal iou
            if gt_index != -1 and pred_index != -2:
                if (gt_labels[gt_index] == labels[pred_index]) and (ioumax > iou_threshold):
                    running_accuracy += 1
        accuracy.append(running_accuracy/len(gt_boxes))
    total_accuracy = mean(accuracy)

    return total_accuracy


def write_detected_boxes(model, data_loader, device, opt, mode = ""):
    model.eval()
    accuracy = []
    for images, targets, img_name in data_loader:
        images = [image.to(device) for image in images]
        # targets = [target.to(device) for target in targets]
        running_accuracy = 0

        with torch.no_grad():
            model = model.cuda()
            pred = model(images)

        labels = list(pred[0]['labels'].cpu().numpy())
        boxes = list(pred[0]['boxes'].detach().cpu().numpy())
        scores = list(pred[0]['scores'].detach().cpu().numpy())
        bbox_file = open(opt.txt_path + mode + "detections/" + img_name[0] + ".txt", "w")
        for i, pred_box in enumerate(boxes):
            label = labels[i]
            score = scores[i]
            x, y, x2, y2 = pred_box
            line = [f"{label} {score} {x} {y} {x2-x} {y2-y}\n"]
            bbox_file.writelines(line)


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
    header = 'Test:'

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

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
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
