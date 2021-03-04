import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image


class TestDataset(object):
    def __init__(self, root, image_paths, transforms=None, augmentations=None):
        self.root = root
        # self.boxes_file = pd.read_csv(boxes_path)
        self.image_paths = image_paths
        self.transforms = transforms
        self.augmentations = augmentations

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = Path(img_path)
        img_name = img_name.stem
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        # boxes = []
        # labels = []
        # select_rows = self.boxes_file.loc[self.boxes_file['Index'] == idx]
        # num_boxes = select_rows['Index'].count()
        # for index, row in select_rows.iterrows():
        #     x = row['X']
        #     y = row['Y']
        #     w = row['W']
        #     h = row['H']
        #     label = row['Label']
        #     boxes.append([x, y, x + w, y + h])
        #     labels.append(label)
        if self.augmentations is not None:
            augmentations = self.augmentations(image=img)
            img = augmentations['image']
        # labels = np.asarray(labels)
        # labels = torch.from_numpy(labels.astype('int64'))  # original: 'long'
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        # iscrowd = torch.zeros(num_boxes, dtype=torch.int64)
        # target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        target = None
        if self.transforms is not None:
              # transforms require target
            img = self.transforms(img, target)
        return img, target, img_name

    def __len__(self):
        return len(self.image_paths)
