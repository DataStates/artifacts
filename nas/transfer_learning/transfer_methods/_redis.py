from ._base import TransferMethod
from ._common import standardize_names
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Optional, Any
import tensorflow as tf
import os
import json
import redis
import redis.lock as rlock
import time
import sys
import subprocess as sp


@contextmanager
def with_rlock(_locktype, redis_conn, name):
    with rlock.Lock(redis_conn, name) as lk:
        yield lk


@contextmanager
def with_rwlock(locktype, redis_conn, name):
    if locktype == "reader":
        # pre-read
        with rlock.Lock(redis_conn, name + ":r"):
            num_readers = redis_conn.incr(name + ":b")
            if num_readers == 1:
                lk = rlock.Lock(redis_conn, name + ":g")
                lk.acquire(token=b"writer")

        yield None

        # post-read
        with rlock.Lock(redis_conn, name + ":r"):
            num_readers = redis_conn.decr(name + ":b")
            if num_readers == 0:
                lk = rlock.Lock(redis_conn, name + ":g")
                lk.local.token = b"writer"
                lk.release()

    elif locktype == "writer":
        with rlock.Lock(redis_conn, name + ":g"):
            yield None
    else:
        raise NotImplementedError()


# metadata schema:
# models: set[id] -- set of id's that can be queried
# model:lock:id -- mutex for storage to HDF5 file since we can't trust POSIX locking on Theta
#                  We hold this lock whenever we write to the file system.
#                  We know that no readers can access this file system storage because to acquire this lock
#                  refcount either moves:
#                   0->1 on creation: in which case there can be no readers because we aren't published yet
#                   1->0 on deletion: and we hold the metadata_lock for writing which prevents readers and writers
# model:refcount:id -- refcount for a model, used to keep track of references
# model:uses:id -- counts the number of times that a model was transferred
# model:prefix:id -- list of model storage layers
# model:graph:id -- graph of the model stored as JSON, only stored when debug=True

class TransferHDF5(TransferMethod):
    def __init__(
        self,
        host: str = "redis://localhost",
        hosts: List[str] = None,
        bulk_storage_path: Path = Path(os.environ.get("TMPDIR", "/tmp")),
        debug=False,
        lock_class=with_rwlock,
        **kwargs
    ):
        super().__init__()
        self.bulk_storage_path = bulk_storage_path
        self._server = None
        if hosts is None:
            self.url = host
        else:
            self.url = hosts[0]

        self.debug = debug
        self.lock_class = lock_class
        if self.debug:
            print(vars(self))

    @staticmethod
    def startup_server(*args, **kwargs) -> Tuple[Any, List[str]]:
        print("starting up redis", flush=True)
        workcomm, hosts = TransferMethod.startup_server()
        print("hosts ", hosts, flush=True)
        attempts = 0
        done = False
        while (not done) and attempts < 5:
            try:
                print("attempting to connect to Redis", file=sys.stderr, flush=True)
                r = redis.Redis.from_url(hosts[0])
                r.scard("models")
                done = True
            except redis.exceptions.BusyLoadingError:
                print("detected redis is NOT ready", file=sys.stderr, flush=True)
                attempts += 1
                time.sleep(5)
        print("detected redis is ready", file=sys.stderr, flush=True)

        return workcomm, hosts

    def teardown_server(self, workcomm):
        if workcomm.rank == 0:
            self.metadata_db.shutdown()
        super().teardown_server()



    @property
    def metadata_db(self):
        if self._server == None:
            self._server = redis.Redis.from_url(self.url)
            # we want to disable creating dumps and saves to avoid cross contamination between experiments
            self._server.config_set("save", "")
            self._server.config_set("appendonly", "no")
        return self._server

    def transfer(
        self, model: tf.keras.Model, id: str, hint=None
    ) -> Tuple[tf.keras.Model, List[str]]:
        # best match ensures that the refcount on the model is incremented if it exists
        best_match, transferred = self._best_match(model)
        if best_match is not None:
            path = self.bulk_storage_path / (best_match + ".h5")
            try:
                model.load_weights(path, by_name=True, skip_mismatch=True)
            except ValueError as e:
                print(best_match, transferred)
                print(model.summary(), model.get_config())
                #print(new_model.summary(), model.get_config())
                raise e

            # decrement the refcount, deleting if needed
            #self.retire_model(best_match)
        return transferred, best_match

    def store(self, id: str, model: tf.keras.Model, prefix: List[str], val_acc: float) -> str:
        suffix = [layer.name for layer in model.layers if layer.name not in prefix]
        if not suffix:
            return False
        if self.debug:
            print("storing ", id)
            self.metadata_db.set(
                "model:graph:" + id, json.dumps(model.get_config()).encode()
            )
        # first we need to ensure that only one writer tries to write the file
        with self.lock_class("writer", self.metadata_db, "metadata_lock"):
            if self.metadata_db.incr("model:refcount:" + id) == 1:
                if self.debug:
                    print("storing:model_new ", id)
                do_store = True
            else:
                if self.debug:
                    print("storing:model_exists ", id)
                do_store = False

        if do_store:
            # we need to save the weights first and ensure that they are written out first
            # it is safe to release the writer lock while writing this because
            # 1. we won't find the model in _best_match because it hasn't been added to the models set yet
            # 2. if another store comes in for the same id, they won't trigger a publish or a write
            # 3. if a store comes in for a model that is in os.unlink in retire_model, we protect this with a model
            #        specific lock, which won't hold up metadata access
            with rlock.Lock(self.metadata_db, "model:lock:" + id):
                model.save_weights(self.bulk_storage_path / (id + ".h5"))
                os.sync()
            # next add them to the metadata server once the file exists; grab the lock
            # to protect the integrity of models
            with self.lock_class("writer", self.metadata_db, "metadata_lock"):
                pipe = self.metadata_db.pipeline()
                pipe.sadd("models", id)
                pipe.rpush("model:prefix:" + id, *[layer.name for layer in model.layers])
                pipe.execute()
        return do_store

    def retire_model(self, id: str):
        # if we actually remove a file,
        # we remove the metadata from the server before we remove the file
        # we then remove the actual file after releasing the lock since no
        # client can find the file now
        retired = False
        storage_lock = None
        with self.lock_class("writer", self.metadata_db, "metadata_lock"):
            if self.metadata_db.decr("model:refcount:" + id) == 0:
                retired = True
                pipe = self.metadata_db.pipeline()
                pipe.srem("models", id)
                pipe.delete("model:refcount:" + id)
                pipe.delete("model:prefix:" + id)
                pipe.execute()
                storage_lock = rlock.Lock(self.metadata_db, "model:lock:" + id)
                storage_lock.acquire()
        if retired:
            if self.debug:
                print("retiring ", id)
            os.unlink(self.bulk_storage_path / (id + ".h5"))
            os.sync()
            storage_lock.release()

    def retain(self, parent_id: str, child_id: str):
        with self.lock_class("writer", self.metadata_db, "metadata_lock"):
            self.metadata_db.incr("model:refcount:" + parent_id)

    def _best_match(self, model: tf.keras.Model) -> Tuple[Optional[str], List[str]]:
        # TODO search metadata in a distributed key-value store
        # alternatively load every model and look for the longest match
        best = None
        best_prefix = []
        best_match = 0
        model_prefix = {layer.name.encode() for layer in model.layers}

        # unfortunately we have to do atomic operations that span several transactions here
        # therefore we hold a lock to ensure that models are retired while we are computing
        # the list of prefixes
        with self.lock_class("reader", self.metadata_db, "metadata_lock"):
            # get the list of candiate models
            models = self.metadata_db.smembers("models")

            # aquire their prefixes
            pipe = self.metadata_db.pipeline()
            for candiate in models:
                key = "model:prefix:" + candiate.decode()
                pipe.lrange(key, 0, -1)
            prefixs = pipe.execute()

            # find a match with at least 1 layer
            for candiate, prefix in zip(models, prefixs):
                prefix_s = set(prefix)
                match = prefix_s & model_prefix
                if len(match) > best_match:
                    best_prefix = [i.decode() for i in prefix]
                    best_match = len(match)
                    best = candiate.decode()

            # if we found a best model, keep it alive
            if best is not None:
                pipe = self.metadata_db.pipeline()
                pipe.incr("model:refcount:" + best)
                pipe.incr("model:uses:" + best)
                pipe.execute()

        if best is not None:
            if self.debug:
                print(f"transferring {best_match} layers from {best}")
        return best, best_prefix
