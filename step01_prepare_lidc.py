from pathlib import Path

import numpy as np
import pylidc as pl
import pickle
import gc
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

class Lidc:
    images = []
    annotations = []

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_images():
    log = open('log.txt', 'a')
    total_images = 6858

    #files = Path('/media/felipe/Blade 14/Dissertacao/Datasets/Processed-LIDC-IDRI').glob('lidc_image_*')
    #total_images = 0
    #for file in files:
    #    total_images = total_images + 1

    print(total_images)

    #files = Path('/media/felipe/Blade 14/Dissertacao/Datasets/Processed-LIDC-IDRI').glob('lidc_image_*')
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_to = total_images*0.8
    #log.write('++++++ List of Images +++++')
    for i in range(0, total_images):
        image_filename = '/media/felipe/Blade 14/Dissertacao/Datasets/Compact-LIDC-IDRI/lidc_image_'+str(i)+'.pkl'
        with open(image_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if len(x_train) < train_to:
                #log.write('TR - ' + image_filename+'\r\n')
                x_train.append(pickle.load(filePointer))
            else:
                #log.write('TS - ' + image_filename+'\r\n')
                x_test.append(pickle.load(filePointer))

        bbox_filename = '/media/felipe/Blade 14/Dissertacao/Datasets/Compact-LIDC-IDRI/lidc_scaled_box_'+str(i)+'.pkl'
        with open(bbox_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if(len(y_train) < train_to):
                #log.write('TR - ' + bbox_filename+'\r\n')
                y_train.append(pickle.load(filePointer))
            else:
                #log.write('TS - ' + bbox_filename+'\r\n')
                y_test.append(pickle.load(filePointer))

    #files = Path('/media/felipe/Blade 14/Dissertacao/Datasets/Processed-LIDC-IDRI').glob('lidc_scaled_box_*')

    #log.write('++++++ List of BBoxes +++++')
    #for file in files:
    #    trainTo = total_images*0.8
    #    with open(file, 'rb') as filePointer:  # Overwrites any existing file.
    #        if(len(y_train) < trainTo):
    #            #log.write('TR - ' + file.name+'\r\n')
    #            y_train.append(pickle.load(filePointer))
    #        else:
    #            #log.write('TS - ' + file.name+'\r\n')
    #           y_test.append(pickle.load(filePointer))

    (xtest), (ytest) = (
        np.asarray(x_test),
        np.asarray(y_test),
    )

    del x_test, y_test

    # Convert the list to numpy array, split to train and test dataset
    print('Convert ytrain to np')
    ytrain = np.asarray(y_train)

    print('GC y_train')
    del y_train
    gc.collect()

    print('Convert xtrain to np')
    xtrain = np.asarray(x_train)

    print('GC x_train')
    del x_train
    gc.collect()

    log.close()

    return xtrain, ytrain, xtest, ytest

def create_or_load_dataset(load=False, save=False, annotation_size_perc=1, file_name='lidc.pkl'):
    images = []
    annotations = []
    if (load):
        with open(file_name, 'rb') as f:
            lidc = pickle.load(f)
            images = lidc.images
            annotations = lidc.annotations
    else:
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
                print(len(images))

        lidc = Lidc()
        lidc.annotations = annotations
        lidc.images = images
        save_object(lidc, file_name)

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





