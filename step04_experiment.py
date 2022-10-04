import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras

from common import bounding_box_intersection_over_union


def print_results(vit_object_detector, image_size, x_test, y_test):
    import matplotlib.patches as patches

    # Saves the model in current path
    vit_object_detector.save("compact-alpha.h5", save_format="h5")

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

def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "../logs/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback
        ],
    )

    return history
