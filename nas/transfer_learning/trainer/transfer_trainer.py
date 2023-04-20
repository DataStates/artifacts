import tensorflow as tf
from tensorflow.keras import backend as K
from deephyper.nas.trainer import BaseTrainer
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.keras.callbacks import import_callback

from deephyper.nas.run._util import (
    default_callbacks_config,
    setup_data,
    load_config,
    preproc_trainer,
    compute_objective,
    get_search_space,
    HistorySaver
)
from ..search.expiring import _transfer_name, _transfer_name_uuid
from ..transfer_methods import TransferMethod, DataStatesModelRepo 
from ..setup import train_data_global
from ..transfer_methods._common import standardize_names
import time
import numpy as np
import logging
import os
import traceback
import hashlib
import uuid

logger = logging.getLogger(__name__)


def custom_setup_data(config, data):
    add_to_config = True
    if type(data) is tuple:
        if len(data) != 2:
            raise RuntimeError(
                f"Loaded data are tuple, should ((training_input, training_output), (validation_input, validation_output)) but length=={len(data)}"
            )
        (t_X, t_y), (v_X, v_y) = data
        if (
            type(t_X) is np.ndarray
            and type(t_y) is np.ndarray
            and type(v_X) is np.ndarray
            and type(v_y) is np.ndarray
        ):
            input_shape = np.shape(t_X)[1:]
            output_shape = np.shape(t_y)[1:]
        elif (
            type(t_X) is list
            and type(t_y) is np.ndarray
            and type(v_X) is list
            and type(v_y) is np.ndarray
        ):
            # interested in shape of data not in length
            input_shape = [np.shape(itX)[1:] for itX in t_X]
            output_shape = np.shape(t_y)[1:]
        elif (
            type(t_X) is np.ndarray
            and type(t_y) is list
            and type(v_X) is np.ndarray
            and type(v_y) is list
        ):
            # interested in shape of data not in length
            input_shape = np.shape(t_X)[1:]
            output_shape = [np.shape(ity)[1:] for ity in t_y]
        elif (
            type(t_X) is list
            and type(t_y) is list
            and type(v_X) is list
            and type(v_y) is list
        ):
            # interested in shape of data not in length
            input_shape = [np.shape(itX)[1:] for itX in t_X]
            output_shape = [np.shape(ity)[1:] for ity in t_y]
        else:
            raise RuntimeError(
                f"Data returned by load_data function are of a wrong type: type(t_X)=={type(t_X)},  type(t_y)=={type(t_y)}, type(v_X)=={type(v_X)}, type(v_y)=={type(v_y)}"
            )
        if add_to_config:
            config["data"] = {
                "train_X": t_X,
                "train_Y": t_y,
                "valid_X": v_X,
                "valid_Y": v_y,
            }
    elif type(data) is dict:
        if add_to_config:
            config["data"] = data
        if len(data["shapes"][0]) == 1:
            input_shape = data["shapes"][0][f"input_0"]
        else:
            input_shape = [
                data["shapes"][0][f"input_{i}"] for i in range(len(data["shapes"][0]))
            ]
        output_shape = data["shapes"][1]
    else:
        raise RuntimeError(
            f"Data returned by load_data function are of an unsupported type: {type(data)}"
        )

    if (
        output_shape == ()
    ):  # basicaly means data with shape=(num_elements) == (num_elements, 1)
        output_shape = (1,)

    if add_to_config:
        return input_shape, output_shape
    else:
        return input_shape, output_shape, data


# we have to fork this code to introduce model freezing and unfreezing
class TrasferTrainer(BaseTrainer):
    """BaseTrainer that supports freezing and unfreezing"""

    def __init__(self, config, model: tf.keras.Model, transfer_method: TransferMethod):
        self._transfer_method = transfer_method
        self.model = model
        super().__init__(config, model)

    def train(
        self,
        num_epochs: int = None,
        with_pred: bool = False,
        last_only: bool = False,
        parent=None,
    ):
        """Train the model.

        Args:
            num_epochs (int, optional): override the num_epochs passed to init the Trainer.
                Defaults to None, will use the num_epochs passed to init the Trainer.
            with_pred (bool, optional): will compute a prediction after the training and will add
                ('y_true', 'y_pred') to the output history. Defaults to False,
                will skip it (use it to save compute time).
            last_only (bool, optional): will compute metrics after the last epoch only.
                Defaults to False, will compute metrics after each training epoch
                (use it to save compute time).
            parent (List[int], optional): the parent arch_seq

        Raises:
            DeephyperRuntimeError: raised when the ``num_epochs < 0``.

        Returns:
            dict: a dictionnary corresponding to the training.
        """
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        self.init_history()

        if num_epochs > 0:

            # TODO load the transferred model if possible
            model_id = 0
            transferred = []
            if isinstance(self._transfer_method, DataStatesModelRepo):
                model_id = np.uint64(uuid.uuid4().int>>64)
            else:
                model_id = _transfer_name(self.config)
           
            self.model = standardize_names(self.model)
            transfer_begin = time.perf_counter()
            if self.config["to_transfer"]:
                # model weights get modified in place
                transferred, parent_id = self._transfer_method.transfer(
                    self.model, id=model_id, hint=parent
                )

            transfer_end = time.perf_counter()
            if transferred:
                for layer in self.model.layers:
                    if layer.name in transferred:
                        layer.trainable = False
            
            self.model_compile()
            # Instantiate an optimizer to train the model.

            # Prepare the metrics.
            time_start_training = time.time()  # TIMING
            if not last_only:
                logger.info(
                    "Trainer is computing metrics on validation after each training epoch."
                )
                history = self.model.fit(
                    self.dataset_train,
                    verbose=self.verbose,
                    epochs=num_epochs,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch,
                    class_weight=self.class_weights,
                )
            else:
                logger.info(
                    "Trainer is computing metrics on validation after the last training epoch."
                )
                if num_epochs > 1:
                    self.model.fit(
                        self.dataset_train,
                        verbose=self.verbose,
                        epochs=num_epochs - 1,
                        steps_per_epoch=self.train_steps_per_epoch,
                        callbacks=self.callbacks,
                        class_weight=self.class_weights,
                    )
                history = self.model.fit(
                    self.dataset_train,
                    epochs=1,
                    verbose=self.verbose,
                    steps_per_epoch=self.train_steps_per_epoch,
                    callbacks=self.callbacks,
                    validation_data=self.dataset_valid,
                    validation_steps=self.valid_steps_per_epoch,
                    class_weight=self.class_weights,
                )

            time_end_training = time.time()  # TIMING
            model_acc = history.history['acc'][0]
            time_model_store_start = time.time()
            self._transfer_method.store(
                model_id, self.model, transferred, model_acc
            )
            time_model_store_end = time.time()

            self.train_history["training_time"] = (
                time_end_training - time_start_training
            )
            self.train_history["storing_time"] = (
                time_model_store_end - time_model_store_start
            )
            self.train_history["transfer_time"] = transfer_end - transfer_begin
            self.train_history[
                "num_layers_transferred"
            ] = len(transferred)
            self.train_history.update(history.history)

        elif num_epochs < 0:
            raise DeephyperRuntimeError(
                f"Trainer: number of epochs should be >= 0: {num_epochs}"
            )

        if with_pred:
            time_start_predict = time.time()
            y_true, y_pred = self.predict(dataset="valid")
            time_end_predict = time.time()
            self.train_history["val_predict_time"] = (
                time_end_predict - time_start_predict
            )

            self.train_history["y_true"] = y_true
            self.train_history["y_pred"] = y_pred

        del self.model
        return self.train_history


# we have to fork this method to change the underlying training method
def build_transfer_trainer(
    transfer_method,
    trainer_cls,
    history_save_dir,
    num_epochs,
    dat_id=None,
    load_data_mode=None,
):
    global run_transfer_trainer

    """returns a transfer trainer"""

    def run_transfer_trainer(config, history_save_dir="."):

        try:
            tf.keras.backend.clear_session()
            # tf.config.optimizer.set_jit(True)

            # setup history saver
            if config.get("log_dir") is None:
                config["log_dir"] = history_save_dir

            save_dir = os.path.join(config["log_dir"], "save")
            saver = HistorySaver(config, save_dir)
            # saver.write_config()

            saver.write_model(None)

            # GPU Configuration if available
            physical_devices = tf.config.list_physical_devices("GPU")
            try:
                for dev in physical_devices:
                    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
                    tf.config.experimental.set_memory_growth(dev, True)
            except Exception:
                # Invalid device or cannot modify virtual devices once initialized.
                logger.info("error memory growth for GPU device")

            # Threading configuration
            if (
                len(physical_devices) == 0
                and os.environ.get("OMP_NUM_THREADS", None) is not None
            ):

                logger.info("OMP_NUM_THREADS is %s", os.environ.get("OMP_NUM_THREADS"))
                num_intra = int(os.environ.get("OMP_NUM_THREADS"))
                try:
                    tf.config.threading.set_intra_op_parallelism_threads(num_intra)
                    tf.config.threading.set_inter_op_parallelism_threads(2)
                except RuntimeError:  # Session already initialized
                    pass
                tf.config.set_soft_device_placement(True)

            seed = config["seed"]

            if seed is not None:
                np.random.seed(seed)
                tf.random.set_seed(seed)

            load_config(config)

            # Check if we put the training data in ray's object store and want to retrieve it from there
            if load_data_mode == "ray":
                import ray
                if dat_id is not None:
                    train_data = ray.get(dat_id)
                    input_shape, output_shape = custom_setup_data(config, train_data)
                # Else(default mode), read load_data fn from config. This reads the file from disk before every training iteration
            elif load_data_mode == "mpicomm":
                train_data = train_data_global
                input_shape, output_shape = custom_setup_data(config, train_data)
            else:
                input_shape, output_shape = setup_data(config)

            search_space = get_search_space(
                config, input_shape, output_shape, seed=seed
            )

            model_created = False
            try:
                model = search_space.sample(config["arch_seq"])
                model_created = True
            except Exception:
                logger.info("Error: Model creation failed...")
                logger.info(traceback.format_exc())

            if model_created:

                # Setup callbacks
                callbacks = []
                cb_requires_valid = False  # Callbacks requires validation data
                callbacks_config = config["hyperparameters"].get("callbacks")
                if callbacks_config is not None:
                    for cb_name, cb_conf in callbacks_config.items():
                        if cb_name in default_callbacks_config:
                            default_callbacks_config[cb_name].update(cb_conf)

                            # Special dynamic parameters for callbacks
                            if cb_name == "ModelCheckpoint":
                                default_callbacks_config[cb_name][
                                    "filepath"
                                ] = saver.model_path

                            # replace patience hyperparameter
                            if "patience" in default_callbacks_config[cb_name]:
                                patience = config["hyperparameters"].get(
                                    f"patience_{cb_name}"
                                )
                                if patience is not None:
                                    default_callbacks_config[cb_name][
                                        "patience"
                                    ] = patience

                            # Import and create corresponding callback
                            callback = import_callback(cb_name)
                            callbacks.append(
                                callback(**default_callbacks_config[cb_name])
                            )

                            if cb_name in ["EarlyStopping"]:
                                cb_requires_valid = "val" in cb_conf["monitor"].split(
                                    "_"
                                )
                        else:
                            logger.error("%s is not an accepted callback!", cb_name)
        
                
                trainer = trainer_cls(
                    config=config, model=model, transfer_method=transfer_method
                )
                trainer.callbacks.extend(callbacks)

                last_only, with_pred = preproc_trainer(config)
                last_only = last_only and not cb_requires_valid

                history = trainer.train(
                    with_pred=with_pred,
                    last_only=last_only,
                    parent=config.get("parent", None),
                    num_epochs=num_epochs,
                )
                # save history
                saver.write_history(history)
                
                result = compute_objective(config["objective"], history)
            else:
                # penalising actions if model cannot be created
                print("Model could not be created returning -Inf!")
                logger.info("Model could not be created returning -Inf!")
                result = -float("inf")

            if np.isnan(result):
                logger.info("Computed objective is NaN returning -Inf instead!")
                result = -float("inf")
            return result
        except Exception as ex:
            raise Exception(ex) from ex

    return run_transfer_trainer
