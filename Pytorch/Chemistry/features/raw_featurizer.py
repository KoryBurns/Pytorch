from __future__ import division
from __future__ import unicode_literals

from rdkit import Chem
from Pytorch.Chemistry.features import Featurizer


class RawFeaturizer(Featurizer):

  def __init__(self, smiles=False):
    self.smiles = smiles

  def _featurize(self, mol):
    if self.smiles:
      return Chem.MolToSmiles(mol)
    else:
        return mol
