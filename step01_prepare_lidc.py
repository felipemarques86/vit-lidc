import numpy as np
import pylidc as pl
from PIL import Image
from pylidc.utils import consensus
from tensorflow import keras
import matplotlib.pyplot as plt

MIN_BOUND = -1000.0
# MIN_BOUND = -1500.0
MAX_BOUND = 600.0
# MAX_BOUND = 1200.0
PIXEL_MEAN = 0.25

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>(1-PIXEL_MEAN)] = 1.
    image[image<(0-PIXEL_MEAN)] = 0.
    return np.array(255 * image, dtype="uint8")





def prepare_dataset(image_size, annotation_size_perc=1):
    total_images = 0
    annotation_list = pl.query(pl.Annotation)
    annotations_count = int(annotation_list.count() * annotation_size_perc)
    images = []
    annotations = []
    for i in range(0, annotations_count):
        annotation = annotation_list[i]
        annotation_bbox = annotation.bbox()
        vol = annotation.scan.to_volume(verbose=False)

        y0, y1 = annotation_bbox[0].start, annotation_bbox[0].stop
        x0, x1 = annotation_bbox[1].start, annotation_bbox[1].stop

        # Get the central slice of the computed bounding box.
        i, j, k = annotation.centroid
        z = max(int(annotation_bbox[2].stop - k) - 1, 0)
        (w, h) = vol[:, :, int(k)].shape

        image = Image.fromarray(normalize(vol[:, :, int(z)]))
        image = image.resize((image_size, image_size))

        images.append(keras.utils.img_to_array(image))
        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        # apply relative scaling to bounding boxes as per given image and append to list
        annotations.append(scaled_bbox)
        total_images = total_images + 1

    # Convert the list to numpy array, split to train and test dataset
    (xtrain), (ytrain) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (xtest), (ytest) = (
        np.asarray(images[int(len(images) * 0.8):]),
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return xtrain, ytrain, xtest, ytest, total_images, images, annotations





