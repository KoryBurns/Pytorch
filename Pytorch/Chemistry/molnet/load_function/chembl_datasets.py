"""
ChEMBL dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import Pytorch.Chemistry as pc
from Pytorch.Chemistry.molnet.load_function.chembl_tasks import chembl_tasks

logger = logging.getLogger(__name__)


def load_chembl(shard_size=2000,
                featurizer="ECFP",
                set="5thresh",
                split="random",
                reload=True):

  data_dir = pc.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "chembl/" + featurizer + "/" + str(split))

  dataset_path = os.path.join(data_dir, "chembl_%s.csv.gz" % set)
  if not os.path.exists(dataset_path):
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_5thresh.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_sparse.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_test.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_train.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_valid.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_test.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_train.csv.gz'
    )
    pc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_valid.csv.gz'
    )

  logger.info("About to load ChEMBL dataset.")
  if reload:
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return chembl_tasks, all_dataset, transformers

  if split == "year":
    train_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_train.csv.gz" % set)
    valid_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_valid.csv.gz" % set)
    test_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_test.csv.gz" % set)

  # Featurize ChEMBL dataset
  logger.info("About to featurize ChEMBL dataset.")
  if featurizer == 'ECFP':
    featurizer = pc.features.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = pc.features.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = pc.features.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = pc.features.RawFeaturizer()

  loader = pc.data.CSVLoader(
      tasks=chembl_tasks, smiles_field="smiles", featurizer=featurizer)

  if split == "year":
    logger.info("Featurizing train datasets")
    train_dataset = loader.featurize(train_files, shard_size=shard_size)
    logger.info("Featurizing valid datasets")
    valid_dataset = loader.featurize(valid_files, shard_size=shard_size)
    logger.info("Featurizing test datasets")
    test_dataset = loader.featurize(test_files, shard_size=shard_size)
  else:
    dataset = loader.featurize(dataset_path, shard_size=shard_size)
  # Initialize transformers
  logger.info("About to transform data")
  if split == "year":
    transformers = [
        pc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for transformer in transformers:
      train = transformer.transform(train_dataset)
      valid = transformer.transform(valid_dataset)
      test = transformer.transform(test_dataset)
  else:
    transformers = [
        pc.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]
    for transformer in transformers:
      dataset = transformer.transform(dataset)

  if split == None:
    return chembl_tasks, (dataset, None, None), transformers

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'scaffold': pc.splits.ScaffoldSplitter()
  }

  splitter = splitters[split]
  logger.info("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    pc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return chembl_tasks, (train, valid, test), transformers