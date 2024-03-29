import torch
import torchvision
import utils
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import transforms as T
import cv2 as cv
import numpy as np
import albumentations as A
import random
from dataset import FliesDataset
from engine import (
    train_one_epoch,
    get_val_loss,
    evaluate,
    get_accuracy,
    write_detected_boxes,
)

# from BoundingBox import BoundingBox
# from BoundingBoxes import BoundingBoxes
# from Evaluator import *


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def augmentations(x_resize, y_resize):
    return A.Compose(
        [
            A.Resize(x_resize, y_resize, interpolation=cv.INTER_AREA),
            A.Flip(p=0.50),
            A.Rotate(limit=90, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.2
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )


def test_augmentations(x_resize, y_resize):
    return A.Compose(
        [
            A.Resize(x_resize, y_resize, interpolation=cv.INTER_AREA),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def draw_bounding_box_from_dataloader(img, target):
    # category_id_to_name = {1: 'cc', 2: 'bz'}
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
    img = (img[0].numpy() * 255).astype(np.uint8)
    img = np.moveaxis(img, 0, -1)
    img = cv.UMat(img).get()
    for i in range(len(target["boxes"])):
        x1 = int(target["boxes"][i][0])
        y1 = int(target["boxes"][i][1])
        x2 = int(target["boxes"][i][2])
        y2 = int(target["boxes"][i][3])
        cv.rectangle(
            img, (x1, y1), (x2, y2), (0, 255, 0), 1
        )  # Draw Rectangle with the coordinates
        category_id = int(target["labels"][i].numpy())
        class_name = category_id_to_name[category_id]
        ((text_width, text_height), _) = cv.getTextSize(
            class_name, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        cv.putText(
            img,
            class_name,
            org=(x1, y1 - int(0.3 * text_height)),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )  # Write the prediction class
    return img


def plot_loss(loss, val_loss, title, filename, large_scale=False):
    # set font sizes for plt
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure()
    plt.plot(loss, label="training loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend(loc="upper right", frameon=False)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    if large_scale:
        plt.ylim(0, 2)
    else:
        plt.ylim(0, 1)
    plt.savefig(filename)


# main function for training. Reading data from csv file includes image paths and bounding boxes
# for each image. creates separate datasets for train and val. building the model on top of
# fasterRcnn resnet50 network.
# on command line use: --data_directory '../data/' --csv_path '../bounding_boxes.csv'
def main():
    set_seed(42)  # set torch seed
    # load command line options
    opt = utils.parse_flags()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    flies_dir = opt.data_directory
    csv_train = opt.csv_train
    csv_val = opt.csv_val
    csv_test = opt.csv_test
    # flies_dir = '../field_data/'
    # csv_train = '../field_train.csv'
    # csv_val = '../field_val.csv'
    # csv_test = '../field_test.csv'
    num_classes = 12  # 11 classes + bg
    x_resize = opt.image_size  # how to resize image before pushing it to model
    y_resize = opt.image_size
    # train on lab data:
    # dataset_train = FliesDataset(flies_dir + 'train', csv_train, get_transform(), augmentations(x_resize, y_resize))
    # dataset_val = FliesDataset(flies_dir + 'val', csv_val, get_transform(), test_augmentations(x_resize, y_resize))
    # dataset_test = FliesDataset(flies_dir + 'test', csv_test, get_transform(), test_augmentations(x_resize, y_resize))
    # train on field data:
    dataset_train = FliesDataset(
        flies_dir, csv_train, get_transform(), augmentations(x_resize, y_resize)
    )
    dataset_val = FliesDataset(
        flies_dir, csv_val, get_transform(), test_augmentations(x_resize, y_resize)
    )
    dataset_test = FliesDataset(
        flies_dir, csv_test, get_transform(), test_augmentations(x_resize, y_resize)
    )

    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # save samples of train and test datasets with boxes
    test_iter = iter(dataloader_test)
    for i in range(len(dataset_test)):
        img, target, img_name = next(test_iter)
        img_ = draw_bounding_box_from_dataloader(img, target[0])
        plt.imsave(
            opt.results_directory
            + "dataset_samples/"
            + img_name[0]
            + "_test_sample.jpg",
            img_,
        )
    val_iter = iter(dataloader_val)
    for i in range(len(dataset_val)):
        img, target, img_name = next(val_iter)
        img_ = draw_bounding_box_from_dataloader(img, target[0])
        plt.imsave(
            opt.results_directory
            + "dataset_samples/"
            + img_name[0]
            + "_val_sample.jpg",
            img_,
        )
    train_iter = iter(dataloader_train)
    for i in range(len(dataset_train)):
        img, target, img_name = next(train_iter)
        img_ = draw_bounding_box_from_dataloader(img, target[0])
        plt.imsave(
            opt.results_directory
            + "dataset_samples/"
            + img_name[0]
            + "_train_sample.jpg",
            img_,
        )

    # get the model using our helper function
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        progress=True,
        num_classes=num_classes,
        pretrained_backbone=True,
    )
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9)
    if opt.field_train:
        print("loading saved model")
        checkpoint = torch.load(opt.model_load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.90)
    loss_list = []
    loss_classifier_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_box_reg_list = []
    loss_val = []
    loss_classifier_val = []
    loss_box_reg_val = []
    loss_objectness_val = []
    loss_rpn_box_reg_val = []
    train_acc_list = []
    val_acc_list = []

    num_epochs = int(opt.num_epochs)
    max_train_acc = -1
    max_val_acc = -1
    min_t_obj_loss = 100
    min_t_calssification_loss = 100
    min_v_obj_loss = 100
    min_v_calssification_loss = 100
    min_v_loss = 1000

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        (
            loss_classifier,
            loss_box_reg,
            loss_objectness,
            loss_rpn_box_reg,
            loss,
        ) = train_one_epoch(
            model, optimizer, dataloader_train, device, epoch, print_freq=10
        )
        (
            v_loss_classifier,
            v_loss_box_reg,
            v_loss_objectness,
            v_loss_rpn_box_reg,
            v_loss,
        ) = get_val_loss(model, dataloader_val, device)
        if epoch % 5 == 0:
            train_acc = get_accuracy(model, dataloader_train, device)
            val_acc = get_accuracy(model, dataloader_val, device)
        if v_loss < min_v_loss:
            if opt.save_model:
                print(f"save model {opt.save_model}")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "train_acc": train_acc,
                    },
                    opt.model_save_path,
                )
            max_val_acc = val_acc
        if train_acc >= max_train_acc:
            max_train_acc = train_acc
        if loss_objectness < min_t_obj_loss:
            min_t_obj_loss = loss_objectness
        if loss_classifier < min_t_calssification_loss:
            min_t_calssification_loss = loss_classifier
        if v_loss_objectness < min_v_obj_loss:
            min_v_obj_loss = v_loss_objectness
        if v_loss_classifier < min_v_calssification_loss:
            min_v_calssification_loss = v_loss_classifier

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloader_val, device=device)
        loss_list.append(loss)
        loss_classifier_list.append(loss_classifier)
        loss_box_reg_list.append(loss_box_reg)
        loss_objectness_list.append(loss_objectness)
        loss_rpn_box_reg_list.append(loss_rpn_box_reg)
        loss_val.append(v_loss)
        loss_classifier_val.append(v_loss_classifier)
        loss_box_reg_val.append(v_loss_box_reg)
        loss_objectness_val.append(v_loss_objectness)
        loss_rpn_box_reg_val.append(v_loss_rpn_box_reg)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    # print accuracy data
    print(f"train acc: {max_train_acc} | val acc: {max_val_acc}")
    # plot loss info
    plot_loss(
        loss_list,
        loss_val,
        "Loss (sum of losses)",
        opt.results_directory + "loss.png",
        large_scale=True,
    )
    plot_loss(
        loss_classifier_list,
        loss_classifier_val,
        "Classification loss",
        opt.results_directory + "classification_loss.png",
    )
    plot_loss(
        loss_box_reg_list,
        loss_box_reg_val,
        "Bounding box regressor loss",
        opt.results_directory + "bbox_regressor_loss.png",
    )
    plot_loss(
        loss_objectness_list,
        loss_objectness_val,
        "Object/background loss",
        opt.results_directory + "objectness_loss.png",
    )
    plot_loss(
        loss_rpn_box_reg_list,
        loss_rpn_box_reg_val,
        "RPN bounding box regressor loss",
        opt.results_directory + "rpn_loss.png",
    )

    plt.figure()
    plt.plot(train_acc_list, label="Train accuracy")
    plt.plot(val_acc_list, label="Validation accuracy")
    plt.legend(loc="lower right")
    plt.title("Classification accuracy vs epochs")
    plt.savefig(opt.results_directory + "acc.png")
    # write text file with loss and acc data
    bbox_file = open(opt.results_directory + "minimum_loss_log.txt", "w")
    line = (
        f"min train class. loss: {min_t_calssification_loss}  | min train objectness loss: {min_t_obj_loss} | max "
        f"train acc: {max_train_acc} | \n min val class. loss: {min_v_calssification_loss}  | min val objectness"
        f" loss: {min_v_obj_loss} |max val accuracy {max_val_acc} "
    )
    bbox_file.write(line)
    # write text files with detected bounding boxes
    write_detected_boxes(model, dataloader_train, device, opt, "train/")
    write_detected_boxes(model, dataloader_val, device, opt, "val/")
    # write_detected_boxes(model, dataloader_test, device, opt, "test/")

    # make predictions on test images
    # pred_iter = iter(dataloader_test)
    # for i in range(len(dataset_test)):
    #     img, _, img_name = next(pred_iter)
    #     img = (img[0].numpy()*255).astype(np.uint8)
    #     img = np.moveaxis(img, 0, -1)
    #     img = cv.UMat(img).get()
    #     # img = cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)
    #     model = model.eval()
    #     with torch.no_grad():
    #         model = model.cuda()
    #         pred = model([dataset_test[i][0].cuda()])
    #
    #     pred_cls = list(pred[0]['labels'].cpu().numpy())
    #     boxes = list(pred[0]['boxes'].detach().cpu().numpy())
    #     scores = list(pred[0]['scores'].detach().cpu().numpy())
    #     # category_id_to_name = {1: 'cc', 2: 'bz'} # on lab training
    #     # category_id_to_name = {1: 'bz', 4: 'other', 5: 'cc'}
    #     category_id_to_name = {1: 'PEACH-FF', 2: 'S-HOUSE-FLY', 3: 'L-HOUSE-FLY', 4: 'OTHER', 5: 'MEDFLY',
    #                            6: 'SPIDER', 7: 'L-ANT', 8: 'Ants', 9: 'Bee', 10: 'LACEWING ', 11: 'ORIENTAL-FF'}
    #     for j in range(len(boxes)):
    #         if scores[j] > 0.5:
    #             x1 = int(boxes[j][0])
    #             y1 = int(boxes[j][1])
    #             x2 = int(boxes[j][2])
    #             y2 = int(boxes[j][3])
    #             class_name = category_id_to_name[pred_cls[j]]
    #             score = str(format(scores[j], '.2f'))
    #             cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw Rectangle with the coordinates
    #             cv.putText(img, class_name+" "+score, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
    #                        thickness=1)  # Write the prediction class
    #     plt.imsave(opt.results_directory+img_name[0]+'_prediction.jpg', img)


if __name__ == "__main__":
    main()
