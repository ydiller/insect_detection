import os
import numpy as np
import torch
from PIL import Image
import pickle


class FliesDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))

    def __getitem__(self, idx):
        # load images ad masks

        assert(self.boxes[idx] == self.imgs[idx].replace("jpg", "p"))

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        boxes_path = os.path.join(self.root, "boxes", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")
        with open(boxes_path, 'rb') as boxes_file:
            boxes = pickle.load(boxes_file)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor

        # If setname cc, -> cc
        # If setname bz, -> bz. Can be 0 and 1.
        num_boxes = len(boxes)
        assert(num_boxes > 0), ("num_boxes = " + str(num_boxes))

        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if "cc" in boxes_path:
            labels = torch.zeros((num_boxes,), dtype=torch.int64)
        else:
            labels = torch.ones((num_boxes,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        # target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
