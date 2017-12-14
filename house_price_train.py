from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import itertools
import argparse
import shutil
import sys

import tensorflow as tf

#_CSV_COLUMNS = ['full_sq', 'life_sq', 'floor', 'max_floor', 'material', 'build_year', 'num_room', 'kitch_sq', 'state',
#'product_type', 'sub_area', 'price_doc']

#_CSV_COLUMN_DEFAULTS = [[0], [0.0], [0.0], [0.0], [0], [0.0], [0.0], [0.0], [0], [''], [''], [0]]


house = pd.read_csv('house-macro.csv')
#house = house.drop(['timestamp', 'id'], axis = 1)
_CSV_COLUMNS = list(house.columns.get_values())
_CSV_COLUMN_DEFAULTS = []
for col in house.columns:
    if pd.api.types.is_string_dtype(house[col]):
        _CSV_COLUMN_DEFAULTS.append(['']);
    else:
        _CSV_COLUMN_DEFAULTS.append([0.0]);
print(_CSV_COLUMN_DEFAULTS)
print(_CSV_COLUMNS)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/Users/Hongtao/Desktop/Stanford/Fall_2017/CS229/final_project/building_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='deep',
    help="Valid model types: {'deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=6, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/Users/Hongtao/Desktop/Stanford/Fall_2017/CS229/final_project/house-train.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/Users/Hongtao/Desktop/Stanford/Fall_2017/CS229/final_project/house-test.csv',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 21329,
    'validation': 9142,
}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  deep_columns = []
  house_cols = house.drop('price_doc', axis = 1)
  for col in house_cols.columns:
    if pd.api.types.is_string_dtype(house_cols[col]): #is_string_dtype is pandas function
      cate_col = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size= len(house_cols[col].unique()))
      cate_col = tf.feature_column.embedding_column(cate_col, len(house_cols[col].unique()))
      deep_columns.append(cate_col)

    else: #is_numeric_dtype is pandas function
      deep_columns.append(tf.feature_column.numeric_column(col))


  return deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  deep_columns = build_model_columns()
  print('deep col: ', len(deep_columns))
  hidden_units = [25, 15]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  return tf.estimator.DNNRegressor(
  	model_dir=model_dir,
    feature_columns=deep_columns,
    hidden_units=hidden_units,
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.01
    ),
    config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, use_quote_delim = False, field_delim = ";")
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('price_doc')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  print('Finish Reading')
  print('num_features: ', len(features))
  return features, labels


def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))
  y = model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, batch_size = _NUM_EXAMPLES['validation']))
  predictions = list(p["predictions"] for p in itertools.islice(y, _NUM_EXAMPLES['validation']))
  print(len(predictions))
  predictions = np.asarray(predictions)
  np.savetxt('predictions.csv', predictions, delimiter = ',', header = 'prediction')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)