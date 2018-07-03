"""
Tox21 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import Pytorch.Chemistry as pc

logger = logging.getLogger(__name__)


def load_tox21(featurizer='ECFP', split='index', reload=True, K=4):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset

  tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]

  data_dir = pc.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "tox21/" + featurizer + "/" + str(split))
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return tox21_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "tox21.csv.gz")
  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    )

  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = pc.features.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)

  loader = deepchem.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      pc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return tox21_tasks, (dataset, None, None), transformers

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'scaffold': pc.splits.ScaffoldSplitter(),
      'butina': pc.splits.ButinaSplitter(),
      'task': pc.splits.TaskSplitter()
  }
  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
  else:
    train, valid, test = splitter.train_valid_test_split(dataset)
    all_dataset = (train, valid, test)
    if reload:
      pc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)
  return tox21_tasks, all_dataset, transformers
