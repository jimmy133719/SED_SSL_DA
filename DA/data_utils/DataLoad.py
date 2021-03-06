import bisect
import numpy as np
import pandas as pd
import torch
import random
import warnings
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utilities.Logger import create_logger
import config as cfg
from utilities.Transforms import Compose

import pdb

torch.manual_seed(0)
random.seed(0)
logger = create_logger(__name__, terminal_level=cfg.terminal_level)


class DataLoadDf(Dataset):
    """ Class derived from pytorch DESED
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (feat_filenames, labels),
            it should contain these columns :
            "feature_filename"
            "feature_filename", "event_labels"
            "feature_filename", "onset", "offset", "event_label"
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set information (feat_filenames, labels, ...)
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """
    def __init__(self, df, encode_function=None, transform=None, return_indexes=False, in_memory=False):
        self.df = df
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.feat_filenames = df.feature_filename.drop_duplicates()
        self.filenames = df.filename.drop_duplicates()
        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes
        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def get_feature_file_func(self, filename):
        """Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        if not self.in_memory:
            data = np.load(filename).astype(np.float32)
        else:
            if self.features.get(filename) is None:
                data = np.load(filename).astype(np.float32)
                self.features[filename] = data
            else:
                data = self.features[filename]
        return data

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.feat_filenames)
        return length

    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)

        """
        features = self.get_feature_file_func(self.feat_filenames.iloc[index])

        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df.iloc[index]["event_labels"]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == self.filenames.iloc[index]][cols]
                if label.empty:
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns))
        if index == 0:
            logger.debug("label to encode: {}".format(label))
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
        else:
            y = label
        sample = features, y
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a ss_model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)

        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)

        return sample

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def add_transform(self, transform):
        if type(self.transform) is not Compose:
            raise TypeError("To add transform, the transform should already be a compose of transforms")
        transforms = self.transform.add_transform(transform)
        return DataLoadDf(self.df, self.encode_function, transforms, self.return_indexes, self.in_memory)


class ConcatDataset(Dataset):
    """
    DESED to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        prec = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(prec, size))
            prec = size
        return cluster_ind

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        super(MultiStreamBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        indices = self.data_source.cluster_indices
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                indices[i] = np.random.permutation(indices[i])
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
