"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time

import Pytorch.Chemistry as pc
import numpy as np
import pandas as pd


def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  data_dir = pc.utils.get_data_dir()
  pdbbind_dir = os.path.join(data_dir, "pdbbind")
  dataset_dir = os.path.join(pdbbind_dir, "%s_%s" % (subset, feat))

  if not os.path.exists(dataset_dir):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz'
    )
    if not os.path.exists(pdbbind_dir):
      os.system('mkdir ' + pdbbind_dir)
    pc.utils.untargz_file(
        os.path.join(data_dir, 'core_grid.tar.gz'), pdbbind_dir)
    pc.utils.untargz_file(
        os.path.join(data_dir, 'full_grid.tar.gz'), pdbbind_dir)
    pc.utils.untargz_file(
        os.path.join(data_dir, 'refined_grid.tar.gz'), pdbbind_dir)

  return deepchem.data.DiskDataset(dataset_dir), tasks


def load_pdbbind_grid(split="random",
                      featurizer="grid",
                      subset="core",
                      reload=True):
  """Load PDBBind datasets. Does not do train/test split"""
  if featurizer == 'grid':
    dataset, tasks = featurize_pdbbind(feat=featurizer, subset=subset)

    splitters = {
        'index': pc.splits.IndexSplitter(),
        'random': pc.splits.RandomSplitter(),
        'time': pc.splits.TimeSplitterPDBbind(dataset.ids)
    }
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dataset)

    transformers = []
    for transformer in transformers:
      train = transformer.transform(train)
    for transformer in transformers:
      valid = transformer.transform(valid)
    for transformer in transformers:
      test = transformer.transform(test)
  else:
    data_dir = pc.utils.get_data_dir()
    if reload:
      save_dir = os.path.join(
          data_dir, "pdbbind_" + subset + "/" + featurizer + "/" + str(split))

    dataset_file = os.path.join(data_dir, subset + "_smiles_labels.csv")

    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
          subset + "_smiles_labels.csv")

    tasks = ["-logKd/Ki"]
    if reload:
      loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
          save_dir)
      if loaded:
        return tasks, all_dataset, transformers

    if featurizer == 'ECFP':
      featurizer = pc.features.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = pc.features.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = pc.features.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = pc.features.RawFeaturizer()

    loader = pc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    transformers = [
        pc.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]

    for transformer in transformers:
      dataset = transformer.transform(dataset)
    df = pd.read_csv(dataset_file)

    if split == None:
      return tasks, (dataset, None, None), transformers

    splitters = {
        'index': pc.splits.IndexSplitter(),
        'random': pc.splits.RandomSplitter(),
        'scaffold': pc.splits.ScaffoldSplitter(),
        'time': pc.splits.TimeSplitterPDBbind(np.array(df['id']))
    }
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dataset)

    if reload:
      pc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)

    return tasks, (train, valid, test), transformers
