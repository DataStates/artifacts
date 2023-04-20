import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
from . import ops
import datetime
import time
import ctypes

unsigned_np_32 = ctypes.c_uint64(32)
long_32 = ctypes.c_uint64(32)
long_right_32_mask = ctypes.c_uint64(4294967295)
max_signed_uint32 = ctypes.c_uint32(2147483647)
max_signed_int32 = ctypes.c_int32(2147483647)

def unsigned_to_signed(ele):
    if ele.value >= max_signed_uint32.value:
        converted_ele = ele.value - max_signed_uint32.value
        converted_ele = ctypes.c_int32(converted_ele).value
    else:
        converted_ele = max_signed_uint32.value - ele.value
        converted_ele = ctypes.c_int32(converted_ele)
        converted_ele = -1*converted_ele.value
    return converted_ele


def split_32(ele):
    left_32 = ele.value >> long_32.value
    right_32 = ele.value & long_right_32_mask.value
    return ctypes.c_uint32(left_32), ctypes.c_uint32(right_32)

def convert_u64bit_to_32_list(lids):
    lids_32 = []
    for ele in lids:
        left_32, right_32  = split_32(ele)
        signed_left_32 = unsigned_to_signed(left_32)
        signed_right_32 = unsigned_to_signed(right_32)
        lids_32.append(signed_left_32)
        lids_32.append(signed_right_32)
    return lids_32


def save_weights(model, prefix, backend, model_ids, lids, config="",
        include_optimizer=True):
    include_optimizer=False
    if not (isinstance(backend, str)):
        raise TypeError("backend should be a string")
    if not (isinstance(config, str)):
        raise TypeError("config should be a string")
    lids_32 = convert_u64bit_to_32_list(lids)
    model_ids32 = convert_u64bit_to_32_list([model_ids])
    tensors = []
    for l in model.layers:
        if l.name in prefix:
            for w in l.weights:
                tensors.append(w)
    if not tensors:
        return
    ops.checkpoint(backend=backend,config=config,lids=lids_32, model_ids = model_ids32, tensors=tensors)

def load_weights(model, prefix, backend, lowners, model_ids, lids, config="",
        include_optimizer=True):
    if not (isinstance(backend, str)):
        raise TypeError("backend should be a string")
    if not (isinstance(config, str)):
        raise TypeError("config should be a string")
    include_optimizer=False
    lids32 = convert_u64bit_to_32_list(lids)
    lowners32 = convert_u64bit_to_32_list(lowners)
    model_ids32 = convert_u64bit_to_32_list([model_ids])
    tensors = []
    for l in model.layers:
        if l.name in prefix:
            for w in l.weights:
                tensors.append(w)
    if not tensors:
        return
    
    ops.restore(backend=backend,
            config=config,
            lowners=lowners32,
            model_ids=model_ids32,
            lids=lids32,
            tensors=tensors)


class CheckpointCallback(Callback):
    """Generic TMCI checkpoint callback class."""

    def __init__(self, backend, config="",
            frequency={'epoch': 1},
            include_optimizer=True):
        if not (isinstance(backend, str)):
            raise TypeError("backend should be a string")
        if not (isinstance(config, str)):
            raise TypeError("config should be a string")
        self.__backend = backend
        self.__config = config
        self.__frequency = frequency
        self.__include_optimizer = include_optimizer

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if 'epoch' in self.__frequency:
            if epoch % self.__frequency['epoch'] == 0:
                save_weights(self.model,
                        backend=self.__backend,
                        config=self.__config,
                        include_optimizer=self.__include_optimizer)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if 'batch' in self.__frequency:
            if epoch % self.__frequency['batch'] == 0:
                save_weights(self.model,
                        backend=self.__backend,
                        config=self.__config,
                        include_optimizer=self.__include_optimizer)
