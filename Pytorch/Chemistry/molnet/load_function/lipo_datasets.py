"""
Lipophilicity dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import Pytorch.Chemistry as pc

logger = logging.getLogger(__name__)


def load_lipo(featurizer='ECFP', split='index', reload=True, move_mean=True):
  """Load Lipophilicity datasets."""
  # Featurize Lipophilicity dataset
  logger.info("About to featurize Lipophilicity dataset.")
  logger.info("About to load Lipophilicity dataset.")
  data_dir = pc.utils.get_data_dir()
  if reload:
    if move_mean:
      dir_name = "lipo/" + featurizer + "/" + str(split)
    else:
      dir_name = "lipo/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  dataset_file = os.path.join(data_dir, "Lipophilicity.csv")
  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    )

  Lipo_tasks = ['exp']

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return Lipo_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()

  loader = pc.data.CSVLoader(
      tasks=Lipo_tasks, smiles_field="smiles", featurizer=featurizer)
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
    return Lipo_tasks, (dataset, None, None), transformers

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
  return Lipo_tasks, (train, valid, test), transformers
