from ._base import TransferMethod
from typing import List, Tuple, Optional, Any
from pathlib import Path
import copy
import os
import tensorflow as tf
from packaging import version
import deephyper
from deephyper.nas.metrics import r2, acc

if version.parse(deephyper.version) >= version.parse("0.4.0"):
    from deephyper.keras.layers import Padding
else:
    from deephyper.layers import Padding


class TransferSimpleHDF5(TransferMethod):
    def __init__(
        self,
        bulk_storage_path: Path = Path(os.environ.get("TMPDIR", "/tmp")),
        debug=False,
        try_search=False,
    ):
        super().__init__()
        self.bulk_storage_path = bulk_storage_path
        self.debug = debug
        self.try_search = try_search
        self.num_layers_transferred = 0

    def retain(self, parent: str, child: str):
        """transfer retension accomplished via hardlink"""
        try:
            os.link(
                (self.bulk_storage_path / (parent + ".h5")),
                (self.bulk_storage_path / (parent + "-" + child + ".h5")),
            )
        except Exception:
            pass

    @staticmethod
    def startup_server(**kwargs) -> Tuple[List[str], List[Any]]:
        return [], []

    @staticmethod
    def teardown_server(handles: List[Any]):
        pass

    def get_num_layers_transferred(self):
        return self.num_layers_transferred

    def transfer(
        self, model: tf.keras.Model, id: str, hint=None
    ) -> Tuple[tf.keras.Model, List[str]]:
        model = tf.keras.models.clone_model(model)
        self.try_search = False
        if hint is None:
            if self.try_search:
                if self.debug:
                    print("hint is None, trying search")
                best_match_path, transfered = self._best_match(model)
            else:
                if self.debug:
                    print("hint is None, search disabled")
                best_match_path = None
                transfered = []
        else:
            if self.debug:
                print(f"hint {hint} is provided, skipping search")
            best_match_path = self.bulk_storage_path / (hint + "-" + id + ".h5")
            transfered = []
        num_layers_transferred = 0
        if best_match_path is not None:
            try:
                loaded = tf.keras.models.load_model(
                    best_match_path,
                    custom_objects={"r2": r2, "acc": acc, "Padding": Padding},
                    compile=False,
                )
                for model_layer, loaded_layer in zip(model.layers, loaded.layers):
                    if model_layer.__class__ != loaded_layer.__class__:
                        break
                    model_config = copy.deepcopy(model_layer.get_config())
                    loaded_config = copy.deepcopy(loaded_layer.get_config())
                    del model_config["name"]
                    del loaded_config["name"]
                    tmp_model_config = {
                        k: v for k, v in model_config.items() if k != "trainable"
                    }
                    tmp_loaded_config = {
                        k: v for k, v in loaded_config.items() if k != "trainable"
                    }

                    if tmp_loaded_config != tmp_model_config:
                        break
                    model_weights = model_layer.get_weights()
                    loaded_weights = model_layer.get_weights()
                    for mw, lw in zip(model_weights, loaded_weights):
                        if mw.shape != lw.shape:
                            break
                    model_layer.set_weights(loaded_layer.get_weights())
                    num_layers_transferred += 1
                    if hint is not None:
                        transfered.append(model_layer.name)

                self.num_layers_transferred = num_layers_transferred
                return model, transfered
            except Exception as e:
                if self.debug:
                    print("loading failed", e)
                pass

        if hint:
            link_path = self.bulk_storage_path / (hint + "-" + id + ".h5")
            if os.path.exists(link_path):
                os.unlink(link_path)
        self.num_layers_transferred = 0
        return model, []

    def store(self, id: str, model: tf.keras.Model, prefix=None) -> str:
        if self.debug:
            print("storing", id)
        path = self.bulk_storage_path / (id + ".h5")
        model.save(path)
        return str(path)

    def retire_model(self, id: str):
        if self.debug:
            print("retiring", id)
        try:
            (self.bulk_storage_path / (id + ".h5")).unlink()
        except Exception:
            pass

    def _best_match(self, model: tf.keras.Model) -> Tuple[Optional[str], List[str]]:
        best_match = None
        transfered: List[str] = []
        for file in self.bulk_storage_path.glob("./*.h5"):
            if self.debug:
                print(f"considering {file}")
            try:
                loaded = tf.keras.models.load_model(file)
            except Exception as e:
                print("failed to load", e)
                continue
            current_transfer = []
            for model_layer, loaded_layer in zip(model.layers, loaded.layers):
                if self.debug:
                    print(f"considering {loaded_layer.name}")
                if model_layer.__class__ != loaded_layer.__class__:
                    if self.debug:
                        print(f"classes don't match")
                    break
                model_config = copy.deepcopy(model_layer.get_config())
                loaded_config = copy.deepcopy(loaded_layer.get_config())
                del model_config["name"]
                del loaded_config["name"]
                if loaded_config != model_config:
                    if self.debug:
                        print(f"configs don't match")
                    break
                model_weights = model_layer.get_weights()
                loaded_weights = model_layer.get_weights()
                for mw, lw in zip(model_weights, loaded_weights):
                    if mw.shape != lw.shape:
                        if self.debug:
                            print(f"weight shape don't match")
                        break
                if self.debug:
                    print(f"appending layer")
                current_transfer.append(model_layer.name)

            if len(current_transfer) > len(transfered):
                transfered = copy.deepcopy(current_transfer)
                best_match = str(copy.deepcopy(file))
        return best_match, transfered
