import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
from dataset import FliesDataset
import transforms as T
import os


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')

    num_classes = 3 # CC, BZ, bacgkround
    flies_dir = os.path.join(os.getenv("HOME"), 'flies')
    dataset_trn = FliesDataset(flies_dir + "/train/", get_transform(train=True))
    dataset_val = FliesDataset(flies_dir + "/val", get_transform(train=False))
    num_trn = len(dataset_trn)    
    num_val = len(dataset_val)
    print(f"Data train has {num_trn} samples")
    print(f"Data val has {num_val} samples")

    # The data is already froma fixed slpit.  
    #   split the dataset in train and val set
    #   num_val = int(num_samples*0.2)
    #   indices = torch.randperm(num_samples).tolist()
    dataset_trn = torch.utils.data.Subset(dataset_trn, list(range(0, num_trn)))
    dataset_val = torch.utils.data.Subset(dataset_val, list(range(0, num_val)))

    # define training and validation data loaders
    data_loader_trn = torch.utils.data.DataLoader(
        dataset_trn, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    print("Going into training!")
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_trn, device, epoch, print_freq=10)
        lr_scheduler.step()
        # evaluate on the val dataset
        if (epoch % num_epochs) == 0:
            print(f"Evaluating epoch #{epoch}")
        evaluate(model, data_loader_val, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
