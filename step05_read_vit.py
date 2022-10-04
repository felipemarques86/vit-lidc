import cv2
import matplotlib.pyplot as plt
import numpy as np
from step01_prepare_lidc import prepare_dataset
from step03_vit_model import create_vit_object_detector
import time

from common import bounding_box_intersection_over_union



def load_results(vit_object_detector,  x_test, y_test):
    import matplotlib.patches as patches

    vit_object_detector.load("alpha.h5")

    i, mean_iou = 0, 0

    # Compare results for 10 images in the test set
    for input_image in x_test[:10]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im, cmap=plt.cm.gray)
        ax2.imshow(im, cmap=plt.cm.gray)

        # input_image = cv2.resize(
        #     input_image, (image_size, image_size), interpolation=cv2.INTER_AREA
        # )
        input_image = np.expand_dims(input_image, axis=0)
        preds = vit_object_detector.predict(input_image)[0]

        (h, w) = (im).shape[0:2]

        top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)

        bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

        box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        # Create the bounding box
        rect = patches.Rectangle(
            (top_left_x, top_left_y),
            bottom_right_x - top_left_x,
            bottom_right_y - top_left_y,
            facecolor="none",
            edgecolor="red",
            linewidth=1,
        )
        # Add the bounding box to the image
        ax1.add_patch(rect)
        ax1.set_xlabel(
            "Predicted: "
            + str(top_left_x)
            + ", "
            + str(top_left_y)
            + ", "
            + str(bottom_right_x)
            + ", "
            + str(bottom_right_y)
        )

        top_left_x, top_left_y = int(y_test[i][0] * w), int(y_test[i][1] * h)

        bottom_right_x, bottom_right_y = int(y_test[i][2] * w), int(y_test[i][3] * h)

        box_truth = top_left_x, top_left_y, bottom_right_x, bottom_right_y

        mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
        # Create the bounding box
        rect = patches.Rectangle(
            (top_left_x, top_left_y),
            bottom_right_x - top_left_x,
            bottom_right_y - top_left_y,
            facecolor="none",
            edgecolor="red",
            linewidth=1,
        )
        # Add the bounding box to the image
        ax2.add_patch(rect)
        ax2.set_xlabel(
            "Target: "
            + str(top_left_x)
            + ", "
            + str(top_left_y)
            + ", "
            + str(bottom_right_x)
            + ", "
            + str(bottom_right_y)
            + "\n"
            + "IoU"
            + str(bounding_box_intersection_over_union(box_predicted, box_truth))
        )
        i = i + 1

    print("mean_iou: " + str(mean_iou / len(x_test[:10])))
    plt.show()




TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 512
SCAN_COUNT_PERC = 0.02
patch_size = 64  # Size of the patches to be extracted from the input image
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.002
weight_decay = 0.00001
batch_size = 64
num_epochs = 150
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 12
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 12
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers
history = []
num_patches = (IMAGE_SIZE // patch_size) ** 2

x_train, y_train, x_test, y_test, total_images, images, annotations = prepare_dataset(IMAGE_SIZE, SCAN_COUNT_PERC)

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

load_results(vit_object_detector, x_test, y_test)
