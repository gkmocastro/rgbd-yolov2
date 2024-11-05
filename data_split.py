import os
import shutil
import random
from pathlib import Path

def split_dataset(images_folder, labels_folder, output_folder, train_percentage):
    # Define paths for the training and test splits
    train_images_folder = Path(output_folder) / "train" / "images"
    test_images_folder = Path(output_folder) / "test" / "images"
    train_labels_folder = Path(output_folder) / "train" / "labels"
    test_labels_folder = Path(output_folder) / "test" / "labels"
    
    # Create directories for training and test splits
    for folder in [train_images_folder, test_images_folder, train_labels_folder, test_labels_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    # Get all image file names and shuffle them
    image_files = list(Path(images_folder).glob("*"))
    random.shuffle(image_files)

    # Calculate the split index
    train_count = int(len(image_files) * train_percentage)

    # Split image files into training and test sets
    train_images = image_files[:train_count]
    test_images = image_files[train_count:]

    # Helper function to copy images and labels
    def copy_files(file_list, image_dest, label_dest):
        for image_file in file_list:
            # Copy image file
            shutil.copy(image_file, image_dest / image_file.name)
            
            # Copy the corresponding label file if it exists
            label_file = Path(labels_folder) / f"{image_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, label_dest / label_file.name)

    # Copy training images and labels
    copy_files(train_images, train_images_folder, train_labels_folder)

    # Copy test images and labels
    copy_files(test_images, test_images_folder, test_labels_folder)

    print(f"Dataset split completed. Training images: {len(train_images)}, Test images: {len(test_images)}.")


if __name__=="__main__":
    split_dataset(
        images_folder="data/data_object_image_2/training/image_2",
        labels_folder="data/labels",
        output_folder="data/data_split",
        train_percentage=0.8
    )