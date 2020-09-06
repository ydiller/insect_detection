import pandas as pd
from PIL import Image
import torch


class FliesDataset(object):
    def __init__(self, root, boxes_path, transforms=None):
        self.root = root
        self.boxes_file = pd.read_csv(boxes_path)
        self.image_paths = self.boxes_file['File path'].unique()
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        select_rows = self.boxes_file.loc[self.boxes_file['Index'] == idx]
        num_boxes = select_rows['Index'].count()
        for index, row in select_rows.iterrows():
            x = row['X']
            y = row['Y']
            w = row['W']
            h = row['H']
            label = row['Label']
            boxes.append([x, y, x + w, y + h])
            labels.append(label)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(num_boxes, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.boxes_file['File path'].unique())