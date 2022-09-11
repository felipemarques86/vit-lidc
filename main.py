from step01_prepare_lidc import prepare_dataset
from step03_vit_model import create_vit_object_detector
from step04_experiment import run_experiment, print_results
import time

TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 214
SCAN_COUNT_PERC = 0.001
patch_size = 32  # Size of the patches to be extracted from the input images
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.002
weight_decay = 0.00001
batch_size = 64
num_epochs = 150
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers
history = []
num_patches = (IMAGE_SIZE // patch_size) ** 2

x_train, y_train, x_test, y_test, total_images, images, annotations = prepare_dataset(IMAGE_SIZE, SCAN_COUNT_PERC)

start_time = time.perf_counter()

print(total_images)

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

history = run_experiment(
    vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
)

print_results(vit_object_detector, IMAGE_SIZE, x_test, y_test)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
