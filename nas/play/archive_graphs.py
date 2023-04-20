#!/usr/bin/env python

import redis
r = redis.Redis(host="localhost", port=7000)
keys = r.keys("model:graph:*")

pipe = r.pipeline()
for key in keys:
    pipe.get(key)
configs = [i.decode() for i in pipe.execute()]
keys = [k.decode() for k in keys]
configs = zip(keys, configs)

import pandas as pd
df = pd.DataFrame(configs, columns=["model", "config"])
df.to_csv("/tmp/configs.csv")
