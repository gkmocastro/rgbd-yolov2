from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import cv2
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YoloDarknetDataset(Dataset):
    def __init__(self, 
                 images_dir,
                 depth_dir, 
                 labels_dir, 
                 classes=["Cyclist", "Pedestrian", "Car"], 
                 transform=None, 
                 max_boxes=22,
                 model_type="rgb",
                 normalization=False):
        """
        Args:
            images_dir (str or Path): Path to the directory containing images.
            labels_dir (str or Path): Path to the directory containing labels.
            classes (list): List of class names.
            transform (callable, optional): Transform to be applied on an image.
            max_boxes (int): The fixed number of bounding boxes per image (default is 22).
            data_mod (str): Modality of data used [RGB, D, RGBD]
        """
        self.images_dir = Path(images_dir)
        self.depth_dir = Path(depth_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.classes = classes
        self.max_boxes = max_boxes
        self.model_type = model_type
        self.normalization = normalization

        # Gather all image files in the directory
        self.image_files = sorted([p for p in self.images_dir.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])
        # Gather all depth files in the directory
        self.depth_files = sorted([p for p in self.depth_dir.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        target = self._load_labels(label_path)

        # if self.normalization:
        #     rgb_image = Image.open(img_path).convert("RGB") 
        #     depth_image = Image.open(depth_path)#.convert("L") #Convert to grayscale for depth
        #     depth_image = np.array(depth_image).astype(np.float32) / 255.0
        #     depth_image = Image.fromarray((depth_image).astype(np.uint8))  # Convert back to PIL Image
        if self.normalization:
            rgb_image = Image.open(img_path).convert("RGB") 
            depth_image = Image.open(depth_path)#.convert("L") #Convert to grayscale for depth
            
            depth_image = np.array(depth_image)  # Normalize depth values if needed
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
            depth_image = 255 - depth_image  # Invert depth values 
            depth_image = Image.fromarray((depth_image).astype(np.uint8))  # Convert back to PIL Image
        else:
            rgb_image = Image.open(img_path).convert("RGB") 
            depth_image = Image.open(depth_path).convert("L")

        if self.transform:
            img, target = self.transform(rgb_image, depth_image, target)


        # separate boxes and labels for each sample
        boxes = target['boxes']
        labels = target['labels']

        # Pad to max_boxes with (-1, 0, 0, 0, 0) for missing boxes
        num_boxes = len(boxes)
        if num_boxes < self.max_boxes:
            padding = [[-1, 0, 0, 0, 0]] * (self.max_boxes - num_boxes)
            boxes += padding
            labels += [-1] * (self.max_boxes - num_boxes)  # Add -1 for missing label entries

        # Truncate if there are more than max_boxes
        boxes = boxes[:self.max_boxes]
        labels = labels[:self.max_boxes]

        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, {'boxes': boxes, 'labels': labels}

    def _load_labels(self, label_path):
        """
        Load labels from a Darknet format .txt file without converting to pixel coordinates.
        
        Args:
            label_path (Path): Path to the .txt file.
        
        Returns:
            boxes (list of lists): Bounding boxes in normalized coordinates [class_id, x_center, y_center, width, height].
            labels (list): Class labels.
        """
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append(int(class_id))
                    boxes.append([class_id, x_center, y_center, width, height])

        return {'boxes': boxes, 'labels': labels}
