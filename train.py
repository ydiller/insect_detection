import torch
import torchvision
import utils
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import transforms as T
import cv2 as cv
import numpy as np
import albumentations as A
from dataset import FliesDataset
from engine import train_one_epoch, get_val_loss, evaluate


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def augmentations():
    return A.Compose([A.Resize(448, 448, interpolation=cv.INTER_AREA)],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def draw_bounding_box_from_dataloader(img, target):
    category_id_to_name = {1: 'cc', 2: 'bz'}
    img = (img[0].numpy()*255).astype(np.uint8)
    img = np.moveaxis(img, 0, -1)
    img = cv.UMat(img).get()
    for i in range(len(target['boxes'])):
        x1 = int(target['boxes'][i][0])
        y1 = int(target['boxes'][i][1])
        x2 = int(target['boxes'][i][2])
        y2 = int(target['boxes'][i][3])
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw Rectangle with the coordinates
        category_id = int(target['labels'][i].numpy())
        class_name = category_id_to_name[category_id]
        ((text_width, text_height), _) = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv.putText(img, class_name, org=(x1, y1 - int(0.3 * text_height)),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(255, 255, 255),
                    thickness=1, lineType=cv.LINE_AA)  # Write the prediction class
    return img


# main function for training. Reading data from csv file includes image paths and bounding boxes
# for each image. creates separate datasets for train and val. building the model on top of
# fasterRcnn resnet50 network.
# on command line use: --data_directory '../data/' --csv_path '../bounding_boxes.csv'
def main():
    # load command line options
    opt = utils.parse_flags()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    flies_dir = opt.data_directory
    csv_train = opt.csv_train
    csv_val = opt.csv_val
    csv_test = opt.csv_test
    num_classes = 3  # cc + bz
    dataset_train = FliesDataset(flies_dir + 'train', csv_train, get_transform(), augmentations())
    dataset_val = FliesDataset(flies_dir + 'val', csv_val, get_transform(), augmentations())
    dataset_test = FliesDataset(flies_dir + 'test', csv_test, get_transform(), augmentations())
    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    test_iter = iter(dataloader_test)
    for i in range(len(dataset_test)):
        img, target = next(test_iter)
        img_ = draw_bounding_box_from_dataloader(img, target[0])
        plt.imsave(opt.results_directory+'test_sample_'+str(i)+'.jpg', img_)

    # get the model using our helper function
    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00008, momentum=0.9)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.90)
    loss = []
    loss_classifier = []
    loss_box_reg = []
    loss_objectness = []
    loss_rpn_box_reg = []
    loss_val = []
    #loss_val = []
    num_epochs = 100
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        loss_dict, running_loss = train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
        v_loss = get_val_loss(model, dataloader_val, device)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloader_val, device=device)
        loss.append(running_loss / len(dataset_train))
        loss_classifier.append(loss_dict['loss_classifier'].item() / len(dataset_train))
        loss_box_reg.append(loss_dict['loss_box_reg'].item() / len(dataset_train))
        loss_objectness.append(loss_dict['loss_objectness'].item() / len(dataset_train))
        loss_rpn_box_reg.append(loss_dict['loss_rpn_box_reg'].item() / len(dataset_train))
        loss_val.append(v_loss / len(dataset_val))
        plt.plot(loss_val)
        plt.savefig('../loss_val.jpg')
        #loss_val.append(loss_val / len(dataset_train))
    # plot loss info
    plt.figure()
    plt.plot(loss, label="loss")
    plt.plot(loss_classifier, label="Classification loss")
    plt.plot(loss_box_reg, label="Bounding box regressor loss")
    plt.plot(loss_objectness, label="Object/background loss")
    plt.plot(loss_rpn_box_reg, label="RPN bounding box regressor loss")
    plt.plot(loss_val, label="Validation loss")
    plt.legend(loc="upper right")
    plt.title('Loss vs epochs')
    plt.savefig('../loss.jpg')

    print("That's it!")

    # make predictions on test images
    pred_iter = iter(dataloader_test)
    for i in range(len(dataset_test)):
        img, _ = next(pred_iter)
        img = (img[0].numpy()*255).astype(np.uint8)
        img = np.moveaxis(img, 0, -1)
        img = cv.UMat(img).get()
        img = cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)
        model = model.eval()
        with torch.no_grad():
            model = model.cuda()
            pred = model([dataset_test[i][0].cuda()])

        pred_cls = list(pred[0]['labels'].cpu().numpy())
        boxes = list(pred[0]['boxes'].detach().cpu().numpy())
        scores = list(pred[0]['scores'].detach().cpu().numpy())
        category_id_to_name = {1: 'cc', 2: 'bz'}
        for j in range(len(boxes)):
            if scores[j] > 0.7:
                x1 = int(boxes[j][0])
                y1 = int(boxes[j][1])
                x2 = int(boxes[j][2])
                y2 = int(boxes[j][3])
                class_name = category_id_to_name[pred_cls[j]]
                score = str(format(scores[j], '.2f'))
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw Rectangle with the coordinates
                cv.putText(img, class_name+" "+score, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                           thickness=1)  # Write the prediction class
        cv.imwrite(opt.results_directory+'prediction_result_'+str(i)+'.jpg', img)


if __name__ == '__main__':
    main()
