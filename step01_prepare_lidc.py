import numpy as np
import pylidc as pl
import pickle
from pathlib import Path

from PIL import Image

# MIN_BOUND = -1000.0
# # MIN_BOUND = -1500.0
# MAX_BOUND = 600.0
# # MAX_BOUND = 1200.0
# PIXEL_MEAN = 0.25

# def normalize(image):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image>(1-PIXEL_MEAN)] = 1.
#     image[image<(0-PIXEL_MEAN)] = 0.
#     return np.array(255 * image, dtype="uint8")

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def process_dataset(annotation_size_perc=1):
    # images = []
    # annotations = []
    count = 0
    annotation_list = pl.query(pl.Annotation)
    annotations_count = int(annotation_list.count() * annotation_size_perc)
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

        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        for j in range(annotation_bbox[2].start, annotation_bbox[2].stop):
            save_object(scaled_bbox, 'D:\\LIDC-IDRI\\lidc_scaled_box_'+str(count)+'.pkl')
            #annotations.append(scaled_bbox)
            #images.append(vol[:, :, j])
            save_object(vol[:, :, j], 'D:\\LIDC-IDRI\\lidc_image_'+str(count)+'.pkl')
            count = count + 1
            print(count)

def load_images():
    files = Path('D:\\LIDC-IDRI').glob('lidc_image_*')
    images = []
    for file in files:
        with open(file, 'rb') as filePointer:  # Overwrites any existing file.
            images.append(pickle.load(filePointer))

    files = Path('D:\\LIDC-IDRI').glob('lidc_scaled_box_*')
    annotations = []
    for file in files:
        with open(file, 'rb') as filePointer:  # Overwrites any existing file.
            annotations.append(pickle.load(filePointer))

    # Convert the list to numpy array, split to train and test dataset
    (xtrain), (ytrain) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (xtest), (ytest) = (
        np.asarray(images[int(len(images) * 0.8):]),
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return xtrain, ytrain, xtest, ytest, images, annotations

def prepare_dataset(annotation_size_perc=1):
    images = []
    annotations = []
    annotation_list = pl.query(pl.Annotation)
    annotations_count = int(annotation_list.count() * annotation_size_perc)
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

        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        for j in range(annotation_bbox[2].start, annotation_bbox[2].stop):
            annotations.append(scaled_bbox)
            images.append(vol[:, :, j])

    # Convert the list to numpy array, split to train and test dataset
    (xtrain), (ytrain) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (xtest), (ytest) = (
        np.asarray(images[int(len(images) * 0.8):]),
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return xtrain, ytrain, xtest, ytest, images, annotations





