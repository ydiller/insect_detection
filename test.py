import cv2 as cv
import torch
import os
import numpy as np
import transforms as T
import albumentations as A
import utils
import matplotlib.pyplot as plt
from dataset import FliesDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from engine import write_field_detected_boxes


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def augmentations(x_resize, y_resize):
    return A.Compose([
        A.Resize(x_resize, y_resize, interpolation=cv.INTER_AREA),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def drawbox_from_prediction(img, boxes, scores, labels):
    category_id_to_name = {1: 'PEACH-FF', 2: 'S-HOUSE-FLY', 3: 'L-HOUSE-FLY', 4: 'OTHER', 5: 'MEDFLY',
                           6: 'SPIDER', 7: 'L-ANT', 8: 'Ants', 9: 'Bee', 10: 'LACEWING ', 11: 'ORIENTAL-FF'}
    category_id_to_color = {1: (255, 0, 0), 2: (255, 128, 0), 3: (255, 255, 0), 4: (128, 255, 0), 5: (0, 255, 0),
                            6: (0, 255, 128), 7: (0, 255, 255), 8: (0, 128, 255), 9: (0, 0, 255), 10: (127, 0, 255),
                            11: (255, 0, 255)}
    # category_id_to_name = {1: 'MEDFLY', 2: 'PEACH-FF'}  # temporary dict for lab dataset
    # category_id_to_color = {1: (0, 255, 0), 2: (0, 0, 255)}
    for j in range(len(boxes)):
        if scores[j] > 0.7:
            x1 = int(boxes[j][0])
            y1 = int(boxes[j][1])
            x2 = int(boxes[j][2])
            y2 = int(boxes[j][3])
            class_name = category_id_to_name[labels[j]]
            color = category_id_to_color[labels[j]]
            score = str(format(scores[j], '.2f'))
            cv.rectangle(img, (x1, y1), (x2, y2), color, 1)  # Draw Rectangle with the coordinates
            cv.putText(img, class_name, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
                       thickness=1)  # Write the prediction class
    return img


def main():
    opt = utils.parse_flags()
    dr = opt.data_directory
    x_resize = 896  # how to resize image before pushing it to model
    y_resize = 896
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 12  # 11 classes + bg
    dataset_test = FliesDataset(dr, opt.csv_test, get_transform(), augmentations(x_resize, y_resize))
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.00008, momentum=0.9)
    checkpoint = torch.load(opt.model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    train_acc = checkpoint['train_acc']
    # model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    # pred_iter = iter(dataloader_test)
    # for i in range(len(dataset_test)):
    #     img, target, img_name = next(pred_iter)
    #     print(img_name)
    #     img = (img[0].numpy()*255).astype(np.uint8)
    #     img = np.moveaxis(img, 0, -1)
    #     img = cv.UMat(img).get()
    #     # img = cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)
    #     with torch.no_grad():
    #         model = model.cuda()
    #         pred = model([dataset_test[i][0].cuda()])
    #     labels = list(pred[0]['labels'].cpu().numpy())
    #     boxes = list(pred[0]['boxes'].detach().cpu().numpy())
    #     scores = list(pred[0]['scores'].detach().cpu().numpy())
    #     pred_img = drawbox_from_prediction(img, boxes, scores, labels)
    #     # cv.imwrite(opt.results_directory + img_name[0] + '_detection.jpg', pred_img)
    #     plt.imsave(opt.results_directory + img_name[0] + '_detection.jpg', pred_img)

    write_field_detected_boxes(model, dataloader_test, device, opt, "test/")

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()