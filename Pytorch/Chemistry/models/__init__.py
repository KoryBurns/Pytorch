"""
Gathers all models in one place for convenient imports
"""
from __future__ import division
from __future__ import unicode_literals

from Pytorch.Chemistry.models.models import Model
from Pytorch.Chemistry.models.sklearn_models import SklearnModel
from Pytorch.Chemistry.models.xgboost_models import XGBoostModel
from Pytorch.Chemistry.models.multitask import SingletaskToMultitask

from Pytorch.Chemistry.models.tensorgraph.tensor_graph import TensorGraph
from Pytorch.Chemistry.models.tensorgraph.fcnet import MultitaskRegressor
from Pytorch.Chemistry.models.tensorgraph.fcnet import MultitaskClassifier
from Pytorch.Chemistry.models.tensorgraph.fcnet import MultitaskFitTransformRegressor
from Pytorch.Chemistry.models.tensorgraph.IRV import TensorflowMultitaskIRVClassifier
from Pytorch.Chemistry.models.tensorgraph.robust_multitask import RobustMultitaskClassifier
from Pytorch.Chemistry.models.tensorgraph.robust_multitask import RobustMultitaskRegressor
from Pytorch.Chemistry.models.tensorgraph.progressive_multitask import ProgressiveMultitaskRegressor, ProgressiveMultitaskClassifier
from Pytorch.Chemistry.models.tensorgraph.models.graph_models import WeaveModel, DTNNModel, DAGModel, GraphConvModel, MPNNModel
from Pytorch.Chemistry.models.tensorgraph.models.symmetry_function_regression import BPSymmetryFunctionRegression, ANIRegression
from Pytorch.Chemistry.models.tensorgraph.models.scscore import ScScoreModel

from Pytorch.Chemistry.models.tensorgraph.models.seqtoseq import SeqToSeq
from Pytorch.Chemistry.models.tensorgraph.models.gan import GAN, WGAN
from Pytorch.Chemistry.models.tensorgraph.models.text_cnn import TextCNNModel
from Pytorch.Chemistry.models.tensorgraph.sequential import Sequential
from Pytorch.Chemistry.models.tensorgraph.models.sequence_dnn import SequenceDNN

#################### Compatibility imports for renamed TensorGraph models.####################

from Pytorch.Chemistry.models.tensorgraph.models.text_cnn import TextCNNTensorGraph
from Pytorch.Chemistry.models.tensorgraph.models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph
