from unittest import TestCase

from nose.tools import assert_equals
from rdkit import Chem

import Pytorch.Chemistry as pc


class TestOneHotFeaturizer(TestCase):

  def test_featurize(self):
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    featurizer = pc.feat.one_hot.OneHotFeaturizer(pc.feat.one_hot.zinc_charset)
    one_hots = featurizer.featurize(mols)
    untransformed = featurizer.untransform(one_hots)
    assert_equals(len(smiles), len(untransformed))
    for i in range(len(smiles)):
        assert_equals(smiles[i], untransformed[i][0])
