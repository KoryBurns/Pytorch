"""
qm7 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import Pytorch.Chemistry as pc
import scipy.io


def load_qm7_from_mat(featurizer='CoulombMatrix',
                      split='stratified',
                      reload=True,
                      move_mean=True):
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    if move_mean:
      dir_name = "qm7/" + featurizer + "/" + str(split)
    else:
      dir_name = "qm7/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  qm7_tasks = ["u0_atom"]

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return qm7_tasks, all_dataset, transformers

  if featurizer == 'CoulombMatrix':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'
      )
    dataset = scipy.io.loadmat(dataset_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = pc.data.DiskDataset.from_numpy(X, y, w, ids=None)
  elif featurizer == 'BPSymmetryFunction':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'
      )
    dataset = scipy.io.loadmat(dataset_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    y = dataset['T']
    w = np.ones_like(y)
    dataset = pc.data.DiskDataset.from_numpy(X, y, w, ids=None)
  else:
    dataset_file = os.path.join(data_dir, "qm7.csv")
    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv'
      )
    if featurizer == 'ECFP':
      featurizer = pc.features.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = pc.features.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = pc.features.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = pc.features.RawFeaturizer()
    loader = pc.data.CSVLoader(
        tasks=qm7_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': pc.splits.IndexSplitter(),
        'random': pc.splits.RandomSplitter(),
        'stratified':
        pc.splits.SingletaskStratifiedSplitter(task_number=0)
    }

    splitter = splitters[split]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = [
        pc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)
    if reload:
      pc.utils.save.save_dataset_to_disk(
          save_dir, train_dataset, valid_dataset, test_dataset, transformers)

    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer='CoulombMatrix',
                       split='stratified',
                       reload=True,
                       move_mean=True):
  data_dir = pc.utils.get_data_dir()
  dataset_file = os.path.join(data_dir, "qm7b.mat")

  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat'
    )
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = pc.data.DiskDataset.from_numpy(X, y, w, ids=None)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': pc.splits.IndexSplitter(),
        'random': pc.splits.RandomSplitter(),
        'stratified':
        pc.splits.SingletaskStratifiedSplitter(task_number=0)
    }
    splitter = splitters[split]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = [
        pc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)

    qm7_tasks = np.arange(y.shape[1])
    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True):
  """Load qm7 datasets."""
  # Featurize qm7 dataset
  print("About to featurize qm7 dataset.")
  data_dir = pc.utils.get_data_dir()
  dataset_file = os.path.join(data_dir, "gdb7.sdf")

  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb7.tar.gz'
    )
    pc.utils.untargz_file(os.path.join(data_dir, 'gdb7.tar.gz'), data_dir)

  qm7_tasks = ["u0_atom"]
  if featurizer == 'CoulombMatrix':
    featurizer = pc.feat.CoulombMatrixEig(23)
  loader = pc.data.SDFLoader(
      tasks=qm7_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'stratified': pc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      pc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
