# pip install Pillow numpy

import os
from PIL import Image
import numpy as np

# from google.colab import drive

# Mount Google Drive
# drive.mount('/content/drive')
dataset_folder = "/Volumes/GoogleDrive/My Drive/0000000000000_KMUTT/Build_a_Deep_CNN/project/detect_glass/train_data"


def read_images_from_folder(dataset_folder):
    images = []
    labels = []

    # Get a list of subfolders (each representing a class)
    classes = os.listdir(dataset_folder)
    class_to_label = {class_name: idx for idx,
                      class_name in enumerate(classes)}

    for class_name in classes:
        class_folder = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)

            # Read the image using PIL
            try:
                image = Image.open(image_path)
                # Optionally, you can resize the image if needed:
                # image = image.resize((desired_width, desired_height))
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                continue

            # Convert PIL image to numpy array
            image_array = np.array(image)

            # Store the image data and class label
            images.append(image_array)
            labels.append(class_to_label[class_name])

    return images, labels


def save_output_to_text(images, labels, output_file):
    with open(output_file, 'w') as file:
        for idx, (image, label) in enumerate(zip(images, labels)):
            file.write(f"Image {idx + 1}:\n")
            file.write(f"Class Label: {label}\n")
            file.write(f"Image Data:\n{image}\n\n")


if __name__ == "__main__":
    # Replace '/content/drive/MyDrive/path/to/your/dataset_folder' with the actual path to your dataset folder in Google Drive
    path_to_dataset_folder = dataset_folder
    images, labels = read_images_from_folder(path_to_dataset_folder)

    # Now, 'images' is a list containing numpy arrays of image data, and 'labels' is a list containing class numbers.
    # You can use these lists for further processing or analysis.

    # Replace 'output.txt' with the desired output filename
    output_file = 'output.txt'
    save_output_to_text(images, labels, output_file)
