# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import re
from shutil import rmtree
from os import makedirs, rename, listdir
from os.path import join, exists, isfile
import time
import glob
import argparse
from tensorflow.keras import applications
print(tf.__version__)

def parse_args():
    parser = argparse.ArgumentParser(description='This script is used for training the hcpa model using dataset from tfrecord files. See more: python3 dr_hcpa_v2_2024.py -h')
    parser.add_argument('--tfrec_dir', type=str, default='./data/all', help='Directory containing TFRecord files')
    parser.add_argument('--dataset', type=str, default='all', help='Name of the dataset')
    parser.add_argument('--results', type=str, default='./results/all', help='Directory to save results')
    parser.add_argument('--exec', type=int, default=0, help='Execution number')
    parser.add_argument('--img_sizes', type=int, default=299, help='Image sizes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lrate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_thresholds', type=int, default=200, help='Number of thresholds')
    parser.add_argument('--wait_epochs', type=int, default=10, help='Number of epochs to wait')
    parser.add_argument('--show_files', type=bool, default=False, help='Show files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level for training')

    return parser.parse_args()

args = parse_args()
# Accessing arguments
TFREC_DIR = args.tfrec_dir
dataset = args.dataset
results = args.results
exec = args.exec
IMG_SIZES = args.img_sizes
# tune it, dependes on Image, size, TPU or GPU
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
NUM_CLASSES = args.num_classes
lrate = args.lrate
num_thresholds = args.num_thresholds
wait_epochs = args.wait_epochs

IMAGE_SIZE = [IMG_SIZES, IMG_SIZES]
decay = lrate / EPOCHS
kepsilon = 1e-7
# constant to customize output
SHOW_FILES = args.show_files
VERBOSE = args.verbose

'''
Create folder for output.
'''
#if exists(results):
#    rmtree(results)
#makedirs(results)

def detect_hardware():
  try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    print('Running on TPU ', tpu_resolver.master())
  except ValueError:
    tpu_resolver = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

  # Select appropriate distribution strategy
  if tpu_resolver:
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
#     print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
  elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
  elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
  else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')

  return strategy


strategy = detect_hardware()
REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')

# not using metadata (only image, for now)
def read_labeled_tfrecord(example, __return_only_label):
    feature_description = {
        "imagem": tf.io.FixedLenFeature([], tf.string),
        "clinical_features": tf.io.FixedLenFeature([NUM_CLINICAL_FEATURES], tf.float32),
        "retinopatia": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = decode_image(example['imagem'])
    clinical = example['clinical_features']  # Now gets all features as a vector
    label = tf.cast(example['retinopatia'], tf.int32)
    
    return image, label, clinical


def read_unlabeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['imagem'])

    return image

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32)  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

# count # of images in files.. (embedded in file name)
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)

def load_dataset(filenames, labeled=True, ordered=False, return_only_label=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda example: read_labeled_tfrecord(example, __return_only_label=return_only_label))
    # returns a dataset of (image, labels) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset(filenames, _return_only_label=False):
    dataset = load_dataset(filenames, labeled=True, return_only_label=_return_only_label)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE*REPLICAS)
    dataset = dataset.map(lambda img, label, clinical: ((img, clinical), label))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_valid_dataset(filenames, _return_only_label=False):
    dataset = load_dataset(filenames, labeled=True, return_only_label=_return_only_label)
    # dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE*REPLICAS)
    dataset = dataset.map(lambda img, label, clinical: ((img, clinical), label))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.map(read_labeled_tfrecord)
    dataset = dataset.batch(BATCH_SIZE*REPLICAS)
    dataset = dataset.map(lambda img, label, clinical: ((img, clinical), label))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def lr_time_based_decay(epoch, lr):
    if epoch < wait_epochs:
        return lr
    else:
        return lr * 1 / (1 + decay * epoch)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def dataset_to_numpy_util(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for batch in dataset:
        if len(batch) == 2:
            (images, clinical), labels = batch
        else:
            images, labels, clinical = batch
            
        numpy_images = images.numpy()
        numpy_clinical = clinical.numpy()
        numpy_labels = labels.numpy()
        break
    
    return numpy_images, numpy_labels, numpy_clinical

def generate_thresholds(num_thresholds, kepsilon=1e-7):
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds -1) for i in range(num_thresholds -2)
    ]
    return [0.0] + thresholds + [1.0]


thresholds = generate_thresholds(num_thresholds, kepsilon)

def detect_num_features(tfrec_dir):
    """Detect number of clinical features from first TFRecord file"""
    files = sorted(glob.glob(os.path.join(tfrec_dir, 'train-*.tfrec')))
    if not files:
        files = sorted(glob.glob(os.path.join(tfrec_dir, 'test-*.tfrec')))
        if not files:
            return 1
    
    dataset = tf.data.TFRecordDataset([files[0]])
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        return len(example.features.feature['clinical_features'].float_list.value)
    return 1

NUM_CLINICAL_FEATURES = detect_num_features(TFREC_DIR)
print(f"Detected {NUM_CLINICAL_FEATURES} clinical features")

def build_model(dim=299, thresholds=thresholds, num_clinical_features=1):
    image_input = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='image_input')
    clinical_input = tf.keras.layers.Input(shape=(num_clinical_features,), name='clinical_input')

    base = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=image_input)
    x = base(image_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    
    y = tf.keras.layers.Dense(64, activation='relu')(clinical_input)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    combined = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.5)(z)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    model = tf.keras.Model(inputs=[image_input, clinical_input], outputs=output)
    
    for layer in base.layers:
        layer.trainable = False

    for layer in model.layers:
        layer.trainable = True

    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    loss = tf.keras.losses.BinaryCrossentropy()

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='AUC'),
        tf.keras.metrics.SensitivityAtSpecificity(0.95),
        tf.keras.metrics.SpecificityAtSensitivity(0.95),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives()
    ]

    model.compile(optimizer = opt, loss = loss, metrics=METRICS)

    return model

# for others investigations we store all the history
histories = []

# these will be split in folds
num_total_train_files = len(tf.io.gfile.glob(TFREC_DIR + '/train*.tfrec'))
num_total_valid_files = len(tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec'))

print('#### Image Size %i, batch_size %i'%
      (IMG_SIZES, BATCH_SIZE*REPLICAS))
print('#### Epochs: %i' %(EPOCHS))

# CREATE TRAIN AND VALIDATION SUBSETS
TRAINING_FILENAMES = sorted(glob.glob(os.path.join(TFREC_DIR, 'train-*.tfrec')))
VALID_FILENAMES = sorted(glob.glob(os.path.join(TFREC_DIR, 'test-*.tfrec')))
print('Train TFRecord files', len(TRAINING_FILENAMES))
print('Test TFRecord files', len(VALID_FILENAMES))

if SHOW_FILES:
    print('Number of training images', count_data_items(TRAINING_FILENAMES))
    print('Number of validation images', count_data_items(VALID_FILENAMES))
    # print('Number of testing images', count_data_items(files_test))

K.clear_session()

print('#### Model number', exec)
with strategy.scope():
    model = build_model(dim=IMG_SIZES, thresholds=thresholds, num_clinical_features=NUM_CLINICAL_FEATURES)

# callback to save best model for each fold
sv = tf.keras.callbacks.ModelCheckpoint(results +'/'+
    dataset + '-%i.weights.h5' %exec, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch')
csv_logger = tf.keras.callbacks.CSVLogger(results +'/'+ dataset + '-%i.csv'%exec)
lr = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay)

tStart = time.time()
#     with strategy.scope():
history = model.fit(
    get_training_dataset(TRAINING_FILENAMES),
    epochs=EPOCHS,
    callbacks = [sv, lr, csv_logger],
    # validation_split=0.1,
    # steps_per_epoch = count_data_items(TRAINING_FILENAMES)/BATCH_SIZE//REPLICAS,
    validation_data = get_training_dataset(VALID_FILENAMES),
    # validation_steps = count_data_items(VALID_FILENAMES)/BATCH_SIZE//REPLICAS,
    verbose=VERBOSE,
)
tElapsed = round(time.time() - tStart, 1)

# save all histories
histories.append(history)

print(' ')
print('Time (sec) elapsed: ', tElapsed)
print('...')

#     evaluate
model.save(results +'/'+ dataset + '-%i.keras' %exec)

imagem, label, clinical = dataset_to_numpy_util(get_valid_dataset(VALID_FILENAMES), 2000)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

probabilities = model.predict([imagem, clinical], steps=1)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(label, probabilities)
auc_keras = auc(fpr_keras, tpr_keras)
test_y_pred = tf.argmax(probabilities, axis=1)

df = pd.DataFrame(thresholds_keras, columns=['thresholds'])
df.insert(1, 'tpr', tpr_keras)
df.insert(2, 'fpr', fpr_keras)
df.insert(3, 'sens', tpr_keras)
df['spec'] = 1-df['fpr']
df.to_csv(results +'/'+ dataset + '-%i-thresholds.csv'%exec, encoding='utf-8', index=False)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC = {:.4f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(results +'/'+ dataset + '-%i.pdf' %exec, format="pdf", bbox_inches="tight")

train_final_stats = model.evaluate(get_valid_dataset(TRAINING_FILENAMES), verbose=VERBOSE)
valid_final_stats = model.evaluate(get_valid_dataset(VALID_FILENAMES), verbose=VERBOSE)

print(f"Train final stats: {train_final_stats}")
print(f"Valid final stats: {valid_final_stats}")
# map
# Name dataset, Execution, Acurracy, AUC from evaluate, AUC from sklearn, Time elapsed 
print(f"{dataset},{exec},{valid_final_stats[1]},{valid_final_stats[2]},{auc_keras},{tElapsed}")
