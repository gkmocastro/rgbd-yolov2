from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from brambox 

class KITTIDataset(Dataset):
    def __init__(self, img_dir, labels_dir, classes, transform=None):
        self.labels_dir = Path(labels_dir)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_list = list(self.img_dir.glob("*.png"))
        self.classes

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        y_label = 1
        return (img, y_label)

    
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YoloDarknetDataset(Dataset):
    def __init__(self, images_dir, labels_dir, classes, transform=None):
        """
        Args:
            images_dir (str or Path): Path to the directory containing images.
            labels_dir (str or Path): Path to the directory containing labels.
            classes (list): List of class names.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.classes = classes

        # Gather all image files in the directory
        self.image_files = sorted([p for p in self.images_dir.glob('*') if p.suffix in ['.jpg', '.jpeg', '.png']])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply transforms if specified
        if self.transform:
            img = self.transform(img)

        # Load the corresponding label file
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = self._load_labels(label_path)

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
            boxes (list of lists): Bounding boxes in normalized coordinates [x_center, y_center, width, height].
            labels (list): Class labels.
        """
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append(int(class_id))
                    boxes.append([x_center, y_center, width, height])

        return boxes, labels

