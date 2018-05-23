import numpy as np
import functools


def check_dataset(dataset):
    assert isinstance(dataset, np.ndarray), (
        'Dataset must be a 2D numpy array')
    assert dataset.shape[1] >= 2, (
        'Need at leat 2 variables: shape is {}'.format(dataset.shape))


def get_features(dataset, feature_names=None):
    if feature_names:
        len_condition = len(feature_names) == dataset.shape[1]
        assert isinstance(feature_names, list) and len_condition,\
            "number of elements in feature_names\
        and number of features in dataset do not match"
        return {k: v for k, v in enumerate(feature_names)}

    return {k: v for k, v in enumerate(range(dataset.shape[1]))}


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper
