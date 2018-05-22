import numpy as np
import functools


def check_dataset(dataset):
    assert isinstance(dataset, np.array),\
        'Dataset must be a numpy 2D array'


def get_features(dataset, feature_names):
    if feature_names:
        len_condition = len(feature_names.keys) == dataset.shape[1]
        assert isinstance(feature_names, dict) and len_condition,\
            "number of elements in feature_names\
        and number of features in dataset do not match"
        return feature_names

    return {k: v for k, v in enumerate(range(dataset.shape[1]))}


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper
