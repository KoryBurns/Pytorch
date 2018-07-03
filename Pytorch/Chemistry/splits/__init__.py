"""
Gathers all splitters in one place for convenient imports
"""
from __future__ import division
from __future__ import unicode_literals

from Pytorch.Chemistry.splits.splitters import ScaffoldSplitter
from Pytorch.Chemistry.splits.splitters import SpecifiedSplitter
from Pytorch.Chemistry.splits.splitters import IndexSplitter
from Pytorch.Chemistry.splits.splitters import IndiceSplitter
from Pytorch.Chemistry.splits.splitters import RandomGroupSplitter
from Pytorch.Chemistry.splits.task_splitter import merge_fold_datasets
from Pytorch.Chemistry.splits.task_splitter import TaskSplitter
