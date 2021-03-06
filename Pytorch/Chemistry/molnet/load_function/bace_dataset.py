"""
bace dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import torch
import Pytorch.Chemistry as pc
from Pytorch.Chemistry.molnet.load_function.bace_features import bace_user_specified_features

logger = logging.getLogger(__name__)


def load_bace_regression(featurizer='ECFP',
                         split='random',
                         reload=True,
                         move_mean=True):
  """Load bace datasets."""
  # Featurize bace dataset
  logger.info("About to featurize bace dataset.")
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    if move_mean:
      dir_name = "bace_r/" + featurizer + "/" + str(split)
    else:
      dir_name = "bace_r/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  dataset_file = os.path.join(data_dir, "bace.csv")

  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv'
    )

  bace_tasks = ["pIC50"]
  if reload:
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return bace_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()
  elif featurizer == 'UserDefined':
    featurizer = pc.features.UserDefinedFeaturizer(
        bace_user_specified_features)

  loader = pc.data.CSVLoader(
      tasks=bace_tasks, smiles_field="mol", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=8192)
  # Initialize transformers
  transformers = [
      pc.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset, move_mean=move_mean)
  ]

  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return bace_tasks, (dataset, None, None), transformers

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'scaffold': pc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    pc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return bace_tasks, (train, valid, test), transformers


def load_bace_classification(featurizer='ECFP', split='random', reload=True):
  """Load bace datasets."""
  # Featurize bace dataset
  logger.info("About to featurize bace dataset.")
  data_dir = pc.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "bace_c/" + featurizer + "/" + str(split))

  dataset_file = os.path.join(data_dir, "bace.csv")

  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv'
    )

  bace_tasks = ["Class"]
  if reload:
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return bace_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()
  elif featurizer == 'UserDefined':
    featurizer = pc.features.UserDefinedFeaturizer(
        bace_user_specified_features)

  loader = pc.data.CSVLoader(
      tasks=bace_tasks, smiles_field="mol", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=8192)
  # Initialize transformers
  transformers = [
      pc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return bace_tasks, (dataset, None, None), transformers

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'scaffold': pc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    pc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return bace_tasks, (train, valid, test), transformers
