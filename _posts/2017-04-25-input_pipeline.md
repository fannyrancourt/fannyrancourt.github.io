---
layout: post
title: Improving the input pipeline with Keras
---

While most people do not care about the efficiency of their input pipeline, it can affect the efficiency of their research by a magnitude. In this post, I'll show how you can speed up your input pipeline with processes and/or threads.

**DISCLAIMER**
If you can make any of the tests faster, please send me an e-mail! I'm genuinely interested.

### Libraries
I'll be using Keras 2.0.0 with TensorFlow 1.0.1.


```python
import operator
import threading
import time
from functools import reduce

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D

def prod(factors):
    return reduce(operator.mul, factors, 1)

sess = tf.Session()
K.set_session(sess)
```

When using threads/processes you will probably need the GIL at some point! So here's a decorator that make any generator thread-safe. I updated the code from [anandalogy](http://anandology.com/blog/using-iterators-and-generators/) from Python2 to Python3. Thanks to him!

```python
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g
```



Let's create our problem now! Let's say we have images with shape [200,200,3] and the ground truth are images with shape [12,12,80]. Feel free to modify any of the constants below.
For each test, we will start `workers = 10` threads/processes. In every case, the queue will have a size of `queue_size = 20`. To be able to run on pretty much any GPU the batch size will be of 10. `TRAINING = True` is a constant to provide to `Keras` when calling sess.run. This allows models using `BatchNorm` (like `keras.applications.Resnet50`) to run.


```python
TRAINING = True
batch_size = 10
input_batch = [batch_size, 200, 200, 3]
output_batch = [batch_size, 12, 12, 80]
queue_size = 20
workers = 10
```

The main advantage of parallelism is to do multiple things at once. To show the great advantage of multiprocessing, the function to create our input will take more than 1 second to process. In your pipeline, this function would go fetch datas on disk, call a database, stream inside a HDF5 file, etc.

```python
def get_input():
    # Super long processing I/O
    time.sleep(1)
    return np.arange(prod(input_batch)).reshape(input_batch).astype(np.float32), np.arange(
        prod(output_batch)).reshape(output_batch).astype(
        np.float32)
```


For our experiment, using TensorFlow's FIFOQueue will make everything simpler and is the first step to speed up our pipeline. More informations on queues [here](https://www.tensorflow.org/programmers_guide/threading_and_queues).

```python
inp = K.placeholder(input_batch)
inp1 = K.placeholder(output_batch)
queue = tf.FIFOQueue(queue_size, [tf.float32, tf.float32], [input_batch, output_batch])
x1, y1 = queue.dequeue() # Tensors for the input and the ground truth.
enqueue = queue.enqueue([inp, inp1])
```

We then need a model. We'll use VGG16 with an MAE loss trained with the RMSProp optimizer. I set the output of the network to be the same as our output shape `[12,12,80]`. The model is not important for this blog post so feel free to change it.

```python
model = keras.applications.VGG16(False, "imagenet", x1, input_batch[1:])
for i in range(3):
    model.layers.pop()
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]
output = model.outputs[0]  # 12x12
output3 = Conv2D(5 * (4 + 11 + 1), (1, 1), padding="same", activation='relu')(
    output)
cost = tf.reduce_sum(tf.abs(output3 - y1))
optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
sess.run(tf.global_variables_initializer())
```

To use the FIFOQueue we need a function to be used by every thread. It's a simple function that will call our `enqueue_op` with data from our `get_input()` function.

```python
def enqueue_data(coord, enqueue_op):
    while not coord.should_stop():
        inp_feed, inp1_feed = get_input()
        sess.run(enqueue_op, feed_dict={inp: inp_feed, inp1: inp1_feed})
```

Let's see what happens when we do not use parallelism. In this example, we bypass the queue by directly feeding the Tensors `x1,y1`. We will process 30 batches for 10 epochs.

```python
print("No threading")
start = time.time()
for i in (range(10)):  # EPOCH
    for j in range(30):  # Batch
        x, y = get_input()
        optimizer_, s = sess.run([optimizer, queue.size()], feed_dict={x1: x, y1: y, K.learning_phase(): int(
                                                                           TRAINING)})
print("Took : ", time.time() - start)
```

    No threading
    Took :  369.2351338863373

As expected, this takes more than 300 seconds since our `get_input()` function is doing a 1 second sleep. Let's use the FIFOQueue now.

We need to start threads to do the enqueuing. I'm not aware of any way to do this with processes, but any contribution is welcomed! Using the queue will save us one copy from Python to C++ because of the feed_dict. Since the queue is already in C++ we do not need to do the copy. I added some code to empty the queue at the end to let the threads die.



```python
coordinator = tf.train.Coordinator()
threads = [threading.Thread(target=enqueue_data, args=(coordinator, enqueue)) for i in
           range(workers)]
for t in threads:
    t.start()
print("Queue")
start = time.time()
for i in (range(10)):  # EPOCH
    for j in range(30):  # Batch
        optimizer_, s = sess.run([optimizer, queue.size()],
                                 feed_dict={K.learning_phase(): int(TRAINING)})
print("Took : ", time.time() - start)


def clear_queue(queue, threads):
    while any([t.is_alive() for t in threads]):
        _, s = sess.run([queue.dequeue(), queue.size()])


coordinator.request_stop()
clear_queue(queue, threads)

coordinator.join(threads)
print("DONE Queue")
```

    Queue
    Took :  61.429038286209106
    DONE Queue

Great improvement! More than 5x speedup! Let's compare it to the GeneratorEnqueuer which uses processes to feed a queue but is still using the feed_dict. The main drawback of this approach is that we need the GIL so that our generator do not get iterate over by multiple threads at the same time. Also, we need to sleep while the queue is empty.


```python
from keras.engine.training import GeneratorEnqueuer


@threadsafe_generator
def get_generator():
    while True:
        yield get_input()


gen = get_generator()
enqueuer = GeneratorEnqueuer(gen, True)
enqueuer.start(max_q_size=queue_size, workers=workers)
time.sleep(1)
print("Keras enqueuer using multiprocess")
start = time.time()
for i in range(10):  # EPOCH
    for j in range(30):  # Batch
        while not enqueuer.queue.qsize():
            time.sleep(0.5)
        x, y = enqueuer.queue.get()
        optimizer_ = sess.run([optimizer], feed_dict={x1: x, y1: y,
                                                      K.learning_phase(): int(
                                                          TRAINING)})
print("Took : ", time.time() - start)
enqueuer.stop()
```

    Keras enqueuer using multiprocess
    Took :  32.17211365699768

Amazing! We get a 2x speedup from the FIFOQueue and a 10x speedup from the original code!
The GeneratorEnqueuer can also be done with threads (by setting `pickle_safe=False`) let's see how it goes!

```python
enqueuer = GeneratorEnqueuer(gen, False)
enqueuer.start(max_q_size=queue_size, workers=workers)
time.sleep(1)
print("Keras enqueuer using threads")
start = time.time()
for i in (range(10)):  # EPOCH
    for j in range(30):  # Batch
        while not enqueuer.queue.qsize():
            time.sleep(0.5)
        x, y = enqueuer.queue.get()
        optimizer_ = sess.run([optimizer], feed_dict={x1: x, y1: y,
                                                      K.learning_phase(): int(
                                                          TRAINING)})
print("Took : ", time.time() - start)

enqueuer.stop()
```

    Keras enqueuer using threads
    Took :  301.63300943374634

So it's better than nothing, but the GIL is really killing everything.

| Method | Time (s)|
|--------|------|
| No threading | 369 |
| FIFOQueue | 61 |
| Keras multiprocessed | 32 |
| Keras threaded | 301 |


# Conclusion
So we saw how to speed up an input pipeline. While FIFOQueues are a big speed up, multiprocessing can speed up the pipeline even more! TFRecord should soon be supported by Keras and I'll update this post when it's the case. Also, there should be a way to feed a FIFOQueue using multiprocessing.
