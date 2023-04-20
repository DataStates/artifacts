from typing import Optional, List, Tuple, Any
import uuid
import hashlib
import tensorflow as tf
import copy
import collections
import time
import json
import abc
import ctypes
import numpy as np
import tmci.plugins
import tmci.checkpoint
import os
import math
from ._base import TransferMethod
import hashlib
from datetime import datetime

# Init ctypes types
DOUBLE = ctypes.c_double
PDOUBLE = ctypes.POINTER(DOUBLE)
PPDOUBLE = ctypes.POINTER(PDOUBLE)
PPPDOUBLE = ctypes.POINTER(PPDOUBLE)


class DataStatesModelRepo(TransferMethod):
    def __init__(self, hosts=None, store_weights=False, allow_partial_stores=False, **kwargs):
        # Links to libmodel_client.so. This calls cpp-store/client-lib.cpp constructor ()
        # which finds all running servers (by reading the list from a text file and emplaces it onto servers vector
        # It then creates a model_client() instance.
        # Note: make sure this is called only once
        self.store_weights = store_weights
        self.allow_partial_stores = allow_partial_stores
        self.lib = tmci.plugins.load("libdummy.so")
        host_str_b = b"\0".join(h.encode() for h in hosts)
        host_str_len = len(host_str_b)
        self.lib.set_servers(ctypes.c_char_p(host_str_b), ctypes.c_size_t(host_str_len))
        self.ancestors = {}
        self.to_hash = {}
        self.to_name = {}
        self.init_rdma()

    def init_rdma(self):
        edges = [0,1,2,3,1,3,2,4,4,5] 
        cid = ctypes.c_uint64(100000)
        result = self.__uint64_array([0] * len(edges))
        edges = self.__uint64_array(edges)
        res_temp = self.lib.get_prefix(edges, len(edges), ctypes.byref(cid), result)

    @staticmethod
    def hash(s: str) -> int:
        return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big")

    def __uint64_array(self, layer):
        return (ctypes.c_uint64 * len(layer))(*layer)

    def __ubyte_array(self, layer):
        return (ctypes.POINTER(ctypes.c_ubyte) * len(layer))(*layer)

    def __np_as_ubyte(self, array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def __hash_layers(self, model_cfg):
        for layer in model_cfg["layers"]:
            # Two layers are shareable only if their configs match
            
            u = layer["name"]
            if u in self.to_hash:
                continue
            h = int.from_bytes(hashlib.md5(u.encode()).digest()[8:], "big")
            self.to_hash[u] = h
            self.to_name[h] = u

    def __extract_tensor_config(self, model, prefix):
        sizes = []
        layers = []
        if not prefix:
            prefix = [layer.name for layer in model.layers]
            prefix = list(set(prefix))
        for layer in model.layers:
            if layer.name in prefix:
                h = self.to_hash[layer.name]
                num_weights = len(layer.weights)
                for i in range(num_weights):
                    layers.append(ctypes.c_uint64(h + i))
                    shp = layer.weights[i].get_shape()
                    dtype_size = layer.weights[i].dtype.size
                    prod = math.prod(shp) * dtype_size
                    sizes.append(prod)
        return layers, sizes

    def __config_to_edges(self, model_cfg):
        edges = []
        for layer in model_cfg["layers"]:
            u = layer["name"]
            for in_node in layer["inbound_nodes"]:
                v = in_node[0][0]
                edges += [self.to_hash[v], self.to_hash[u]]
        return self.__uint64_array(edges)

    def store_meta(self, id, model_cfg ):
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        lids = lsizes = lowners = self.__uint64_array([id] * len(model_cfg["layers"]))
        return self.lib.store_meta(
            ctypes.c_uint64(id), edges, len(edges), lids, lowners, lsizes, len(lids)
        )

    def _best_match(self, model):
        return self.get_prefix(model.get_config())

    def get_prefix(self, model_cfg):
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        cid = ctypes.c_uint64()
        result = self.__uint64_array([0] * len(edges))
        res_len = self.lib.get_prefix(edges, len(edges), ctypes.byref(cid), result)
        transferred = [self.to_name[result[i]] for i in range(res_len)]
        transferred = list(set(transferred))
        return (cid, transferred)

    def get_time_stamps(self, function_key=None):
        if function_key:
            b_function_key = function_key.encode("utf-8")
            n = self.lib.get_num_time_stamps_by_key(b_function_key)
            assert n != -1, "could not find key"
            c_ts = self.__uint64_array([0] * n)
            res = self.lib.get_time_stamps_by_key(c_ts, b_function_key)
            assert res != -1, "could not find key"
            return c_ts[:]

        n = self.lib.get_num_time_stamps()
        c_ts = self.__uint64_array([0] * n)
        _ = self.lib.get_time_stamps(c_ts)
        return c_ts[:]

    def clear_time_stamps(self, function_key=None):
        if function_key:
            ret = self.lib.clear_time_stamps_by_key(function_key)
            assert ret, "could not find key"

        self.lib.clear_time_stamps()

    def store(self, id: int, model: tf.keras.Model, prefix: List[int], val_acc: float) -> str:
        suffix = [layer.name for layer in model.layers if layer.name not in prefix]
        if not suffix:
            return False
        suffix = list(set(suffix))
        model_config = model.get_config()
        cid = ctypes.c_uint64(id)
        self.__hash_layers(model_config)
        lids, sizes = self.__extract_tensor_config(model, suffix)
        if self.store_weights:
            tmci.checkpoint.save_weights(
                model,
                suffix,
                backend="dummy",
                model_ids=cid,
                lids=lids,
                config=".",
                include_optimizer=False,
            )
        if self.allow_partial_stores:
            lids = self.__uint64_array([lid.value for lid in lids])
            lsizes = self.__uint64_array(sizes)
            lowners = self.__uint64_array([id] * len(lids))
        else:
            if len(suffix) < len(model.layers):
                if id not in self.ancestors:
                    raise Exception("cannot store partial model without ancestor")
                comp = copy.deepcopy(self.ancestors[id])
                for i in range(len(lids)):
                    comp[lids[i].value] = (id, sizes[i])
                lids = self.__uint64_array(list(comp.keys()))
                lsizes = self.__uint64_array([size for _, size in comp.values()])
                lowners = self.__uint64_array([owner for owner, _ in comp.values()])
            else:
                lids = self.__uint64_array([lid.value for lid in lids])
                lsizes = self.__uint64_array(sizes)
                lowners = self.__uint64_array([id] * len(lids))
        edges = self.__config_to_edges(model_config)
        self.lib.store_meta(cid, edges, len(edges), lids, lowners, lsizes, len(lids), ctypes.c_float(val_acc))
        #self.lib.update_ref_counter(cid, 1)
        return True

    def transfer(
        self, model: tf.keras.Model, id: int, hint=None
    ) -> Tuple[tf.keras.Model, List[str]]:
        model_config = model.get_config()
        self.__hash_layers(model_config)
        cid, prefix = self.get_prefix(model_config)
        if not prefix:
            comp = {}
            prefix_lids, sizes = self.__extract_tensor_config(model, prefix)
            for i in range(len(prefix_lids)):
                comp[prefix_lids[i].value] = (id, 0)
            self.ancestors[id] = comp
            return [], None

        lids, sizes = self.__extract_tensor_config(model, prefix)
        lowners = self.__uint64_array([0] * len(lids))
        prefix_lids = self.__uint64_array(lids)
        self.lib.get_composition(cid, prefix_lids, lowners, len(prefix_lids))
        lowners_ctypes = [ctypes.c_uint64(el) for el in lowners[:]]
        tmci.checkpoint.load_weights(
            model,
            prefix,
            backend="dummy",
            lowners=lowners_ctypes,
            model_ids=cid,
            lids=lids,
            config=".",
            include_optimizer=False,
        )
        comp = {}
        for i in range(len(prefix_lids)):
            comp[lids[i].value] = (lowners[i], sizes[i])
        self.ancestors[id] = comp
        return prefix, cid

    def retire_model(self, id):
        id_int = self.hash(id)
        cid = ctypes.c_uint64(id_int).value
        ret = self.lib.update_ref_counter(id_int, -1)
        return ret
    
    def retain(self, parent_id: str, child_id: str):
        pass

    def startup_server(ds_colocated=False, **kwargs) -> Tuple[List[str], List[Any]]:
        return TransferMethod.startup_server(ds_colocated=ds_colocated)

    def teardown_server(self, workcomm):
        if workcomm.rank == 0:
            self.lib.shutdown()
        super().teardown_server()
