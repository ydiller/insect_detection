import os
import cv2 as cv
import torch
import transforms as T
import albumentations as A
import utils
from test_dataset import TestDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from engine import write_test_field_detected_boxes


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def augmentations(x_resize, y_resize):
    return A.Compose([
        A.Resize(x_resize, y_resize, interpolation=cv.INTER_AREA)])


def main():
    opt = utils.parse_flags()
    dr = opt.data_directory
    x_resize = 896  # how to resize image before pushing it to model
    y_resize = 896
    img_paths = []
    for root, dirs, files in os.walk(dr):
        for file in files:
            img_paths.append(root+file)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    num_classes = 12  # 11 classes + bg
    dataset_test = TestDataset(dr, img_paths, get_transform(), augmentations(x_resize, y_resize))
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.00008, momentum=0.9)
    checkpoint = torch.load(opt.model_load_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    train_acc = checkpoint['train_acc']
    model.eval()

    write_test_field_detected_boxes(model, dataloader_test, device, opt, "test/")

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()