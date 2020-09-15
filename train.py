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
from engine import train_one_epoch, evaluate


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
    csv_train = opt.csv_path
    csv_val = opt.csv_path
    num_classes = 3  # cc + bz
    dataset_train = FliesDataset(flies_dir + 'train', csv_train, get_transform(), augmentations())
    dataset_val = FliesDataset(flies_dir + 'train', csv_val, get_transform(), augmentations())
    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    for i in range(len(dataset_train)):
        img, target = next(iter(dataloader_train))
        img_ = draw_bounding_box_from_dataloader(img, target[0])
        plt.imsave(opt.results_directory+'dataloader_sample_'+str(i)+'.jpg', img_)

    # get the model using our helper function
    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
    loss_values = []
    num_epochs = 300
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        running_loss = train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloader_val, device=device)
        loss_values.append(running_loss / len(dataset_train))

    # plot loss info
    # plt.plot(loss_values)
    # plt.savefig('../loss.jpg')

    print("That's it!")

    # test on one image
    # model = model.eval()
    # with torch.no_grad():
    #     model = model.cuda()
    #     pred = model([dataset_train[0][0].cuda()])
    #     print(pred)
    #
    # pred_cls = list(pred[0]['labels'].cpu().numpy())
    # boxes = list(pred[0]['boxes'].detach().cpu().numpy())
    # scores = list(pred[0]['scores'].detach().cpu().numpy())
    # img = cv.imread('../data_samples/train/set-01-cc-0.jpg.jpg')
    # img = cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)
    # category_id_to_name = {1: 'cc', 2: 'bz'}
    # for i in range(len(boxes)):
    #     if scores[i] > 0.7:
    #         x1 = int(boxes[i][0])
    #         y1 = int(boxes[i][1])
    #         x2 = int(boxes[i][2])
    #         y2 = int(boxes[i][3])
    #         class_name = category_id_to_name[pred_cls[i]]
    #         cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw Rectangle with the coordinates
    #         cv.putText(img, class_name, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
    #                    thickness=1)  # Write the prediction class
    # cv.imwrite('../img.jpg', img)


if __name__ == '__main__':
    main()
