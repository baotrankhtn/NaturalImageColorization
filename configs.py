import math

# Train
PATH_DATASET = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/dataset/train/'
PATH_SAVED_MODEL = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/train/model.{epoch:02d}-{loss:.5f}-{val_loss:.5f}.h5'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_WIDTH_SMALL = 128
IMG_HEIGHT_SMALL = 128
BATCH_SIZE = 16
EPOCHS = 40
NUM_SAMPLES = 228942
SPLIT_TRAIN_VAL = 0.9
T = 1.8 # Constant for adjusting saturation
STEPS_PER_EPOCH_TRAIN = math.ceil(NUM_SAMPLES*SPLIT_TRAIN_VAL/BATCH_SIZE)
STEPS_PER_EPOCH_VAL = math.ceil(NUM_SAMPLES*(1-SPLIT_TRAIN_VAL)/BATCH_SIZE)

# Test
PATH_TEST_DATASET = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/dataset/test/input_1/'
PATH_TEST_DATASET_GRAYSCALE = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/dataset/test/input_1_grayscale/'
PATH_TEST_DATASET_PREDICTION = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/dataset/test/T18_output_1/'
PATH_TEST_SAVED_MODEL = '/Volumes/Data/Projects/Python/Projects/[Master]NaturalImageColorization/NaturalImageColorization/saved_models/release/final.model.T18.14-0.02601-0.02661.h5' 