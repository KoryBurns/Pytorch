"""
qm9 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import Pytorch.Chemistry as pc

logger = logging.getLogger(__name__)


def load_qm9(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True):
  """Load qm9 datasets."""
  # Featurize qm9 dataset
  logger.info("About to featurize qm9 dataset.")
  data_dir = pc.utils.get_data_dir()
  if reload:
    if move_mean:
      dir_name = "qm9/" + featurizer + "/" + str(split)
    else:
      dir_name = "qm9/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunction', 'MP', 'Raw']:
    dataset_file = os.path.join(data_dir, "gdb9.sdf")

    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
      )
      pc.utils.untargz_file(
          os.path.join(data_dir, 'gdb9.tar.gz'), data_dir)
  else:
    dataset_file = os.path.join(data_dir, "qm9.csv")
    if not os.path.exists(dataset_file):
      pc.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv'
      )

  qm9_tasks = [
      "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
      "h298", "g298"
  ]

  if reload:
    loaded, all_dataset, transformers = pc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return qm9_tasks, all_dataset, transformers

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunction', 'MP', 'Raw']:
    if featurizer == 'CoulombMatrix':
      featurizer = pc.features.CoulombMatrix(29)
    elif featurizer == 'BPSymmetryFunction':
      featurizer = pc.features.BPSymmetryFunction(29)
    elif featurizer == 'Raw':
      featurizer = pc.features.RawFeaturizer()
    elif featurizer == 'MP':
      featurizer = pc.features.WeaveFeaturizer(
          graph_distance=False, explicit_H=True)
    loader = pc.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)
  else:
    if featurizer == 'ECFP':
      featurizer = pc.features.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = pc.features.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = pc.features.WeaveFeaturizer()
    loader = pc.data.CSVLoader(
        tasks=qm9_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  if split == None:
    raise ValueError()

  splitters = {
      'index': pc.splits.IndexSplitter(),
      'random': pc.splits.RandomSplitter(),
      'stratified':
      pc.splits.SingletaskStratifiedSplitter(task_number=11)
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
  return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers
