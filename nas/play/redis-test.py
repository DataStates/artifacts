#!/usr/bin/env python
import redis
import redis.lock as rd

#copied port from docker container, 6379 is the default
r = redis.Redis(host="localhost", port=40353)

r.keys("prefix:*")

r.keys("*testing*")

r.rpush("testing", *["a", 'b', 'c'])

r.lrange("testing", 0, -1)

r.delete("testing")

r.incr("testing_count")

r.get("testing_count")

r.decr("testing_count")

pipe = r.pipeline()
pipe.lrange("testing", 0, -1)
pipe.lrange("testing", 0, -1)
pipe.execute()


r.srem("myset", "bar")

r.srem("myset", "foo")

r.sadd("myset", "foo")

r.sadd("myset", "bar")

r.smembers("myset")

with rd.Lock(r, "metadata_lock"):
    r.get("testing_count")
