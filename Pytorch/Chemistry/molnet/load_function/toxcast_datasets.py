"""
TOXCAST dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import Pytorch.Chemistry as pc

logger = logging.getLogger(__name__)


def load_toxcast(featurizer='ECFP', split='index', reload=True):

  data_dir = pc.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir,
                            "toxcast/" + featurizer + "/" + str(split))

  dataset_file = os.path.join(data_dir, "toxcast_data.csv.gz")
  if not os.path.exists(dataset_file):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz'
    )

  dataset = pc.utils.save.load_from_disk(dataset_file)
  logger.info("Columns of dataset: %s" % str(dataset.columns.values))
  logger.info("Number of examples in dataset: %s" % str(dataset.shape[0]))
  TOXCAST_tasks = dataset.columns.values[1:].tolist()

  if reload:
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return TOXCAST_tasks, all_dataset, transformers

  # Featurize TOXCAST dataset
  logger.info("About to featurize TOXCAST dataset.")

  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()

  loader = pc.data.CSVLoader(
      tasks=TOXCAST_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers
  transformers = [
      pc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return TOXCAST_tasks, (dataset, None, None), transformers

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

  return TOXCAST_tasks, (train, valid, test), transformers
