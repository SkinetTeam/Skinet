"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import tensorflow as tf

version = tf.__version__
TF_MAJOR, TF_MINOR, TF_PATCH = [int(v) for v in version.split('.')][:3]


def get_version():
    return TF_MAJOR, TF_MINOR, TF_PATCH


############################################################
#  Arguments rewrite
############################################################
def crop_and_resize_v1(image, boxes, box_indices, crop_size, method, extrapolation_value, name):
    return tf.image.crop_and_resize(image, boxes, box_ind=box_indices, crop_size=crop_size, method=method,
                                    extrapolation_value=extrapolation_value, name=name)


def crop_and_resize_v2(image, boxes, box_indices, crop_size, method, extrapolation_value, name):
    return tf.image.crop_and_resize(image, boxes, box_indices=box_indices, crop_size=crop_size, method=method,
                                    extrapolation_value=extrapolation_value, name=name)


############################################################
#  Defining methods to use
############################################################
if TF_MAJOR == 1 and 3 <= TF_MINOR <= 15:
    if TF_MINOR < 14:
        WHERE_FUNC = tf.where
    else:
        WHERE_FUNC = tf.compat.v1.where_v2

    if TF_MINOR < 13:
        CROP_AND_RESIZE_FUNC = crop_and_resize_v1
        INTERSECTION_FUNC = tf.sets.set_intersection
    else:  # TF >= 1.13
        CROP_AND_RESIZE_FUNC = crop_and_resize_v2
        INTERSECTION_FUNC = tf.sets.intersection

    if TF_MINOR < 12:
        TO_DENSE_FUNC = tf.sparse_tensor_to_dense
    else:  # TF >= 1.12
        TO_DENSE_FUNC = tf.sparse.to_dense

    if TF_MINOR < 10:
        LOG_FUNC = tf.log
    else:  # TF >= 1.10
        LOG_FUNC = tf.math.log
else:
    raise NotImplementedError(f"Compatibility with TF {tf.__version__} is not implemented")


def crop_and_resize(image, boxes, box_indices=None, crop_size=None, method='bilinear', extrapolation_value=0,
                    name=None):
    return CROP_AND_RESIZE_FUNC(image, boxes, box_indices, crop_size, method, extrapolation_value, name)


def intersection(a, b, validate_indices=True):
    return INTERSECTION_FUNC(a=a, b=b, validate_indices=validate_indices)


def log(x, name=None):
    return LOG_FUNC(x, name=name)


def to_dense(sp_input, default_value=None, validate_indices=True, name=None):
    return TO_DENSE_FUNC(sp_input=sp_input, default_value=default_value, validate_indices=validate_indices, name=name)


def where(condition, x=None, y=None, name=None):
    return WHERE_FUNC(condition=condition, x=x, y=y, name=name)
