from ._base import TransferMethod
from typing import List, Tuple, Optional, Any
import tensorflow as tf


class TransferNoop(TransferMethod):
    @staticmethod
    def startup_server(**kwargs) -> Tuple[List[str], List[Any]]:
        return [], []

    @staticmethod
    def teardown_server(handles: List[Any]):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def transfer(
        self, model: "tf.keras.Model", id: str, hint=None
    ) -> Tuple["tf.keras.Model", List[str]]:
        return [], id

    def store(self, id: str, model: "tf.keras.Model", prefix: List[str], val_acc: float) -> str:
        """store weights; empty prefix means store everything; returns the id of a model"""
        return True

    def retire_model(self, id: str):
        """removes a model and its weights"""
        pass

    def retain(self, parent: str, child: str):
        """no transfer"""
        pass

    def _best_match(self, model: tf.keras.Model) -> Tuple[Optional[str], List[str]]:
        return None, []
