---
layout: post
title: Ordered multi-processed generator in Keras
---

If you have used Keras extensively, you are probably aware that using
`model.fit_generator(..., pickle_safe=True)` may not do what you expect.
In fact, generators cannot be shared across processes in `Python`. This cause your generator to be copied instead. So your training loop will see the same data over and over again. While this is not that big of a deal in most cases, this cause the function `model.predict_generator` to be useless for `workers` > 1.

## Solving the problem
To solve this problem, I have been toying around with `Pytorch`'s [Dataset](http://pytorch.org/docs/_modules/torch/utils/data/dataset.html#Dataset) to use it into `Keras` code base. In this post, I'll share my solution that I hope, will be merged into `Keras` soon enough.

## Code

Let's start with some `import`, we'll need a `ProcessPoolExecutor` to submit jobs to the different processes. A single thread-safe `queue` is all what we need.

```python
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import cycle
from queue import Queue
from threading import Thread
```

As explained earlier, I used `Pytorch`'s *Dataset* object.

```python
class Dataset():
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

For our tests, we shall create a fake *Dataset*. Here, the `time.sleep(1)` is representing all the preprocessing tasks, reading a file, data augmentation, resizing. For the sake of this post, the dataset is an *echo* dataset. It returns what you give him.

```python
class ExampleDataset(Dataset):
    def __getitem__(self, index):
        time.sleep(1)
        return os.getpid(), index

    def __len__(self):
        return 100
```

The class `MultiProcessExecutor` was made to replace `GeneratorEnqueuer` from `Keras`. While the constructor lacks many arguments, this is just to showcase the power of this object. The `MultiProcessExecutor` will create a `ProcessPoolExecutor` and when there is place in the `queue`, it will submit a task to the executor with [executor.submit](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit).

 Submitting a request will return a `Future` object, you can wait on that object to get the result. The `Future` objects are queued so that they will be read in order.

```python
class MultiProcessExecutor():
    def __init__(self, dataset):
        self.workers = 5
        self.executor = ProcessPoolExecutor(self.workers)
        self.futures = {}
        self.dataset = dataset
        self.queue = Queue(self.workers * 2)
        self.run_thread = None

    def start(self):
        self.run_thread = Thread(target=self.run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def run(self):
        """ This will queue up 2*'workers' tasks in order """
        indexes = cycle(range(len(self.dataset)))
        for i in indexes:
            self.queue.put(self.executor.submit(self.dataset.__getitem__, [i]), block=True)

    def get_item(self):
        while True:
            yield self.queue.get(block=True).result()
```

# Test
To test our executor, we will ask for 100 items while timing it. `executor.get_item()` is returning a generator that will wait for the queue and will return the `Future`'s result.


```python
dataset = ExampleDataset()
executor = MultiProcessExecutor(dataset)
executor.start()
getter = executor.get_item()
start = time.time()
for i in range(100):
    result = next(getter)
print("Took executor",time.time()-start)
```

    Took executor 20.045644760131836

Pretty much what we expected. 5 workers doing a 100 second task should reduce the time by 5.

Now, let's compare it to `Keras`'s `GeneratorEnqueuer`. We'll create a generator that is doing exactly the same thing as our `ExampleDataset`.

```python
from keras.engine.training import GeneratorEnqueuer
def keras_gen():
    while True:
        time.sleep(1)
        yield os.getpid()
```



```python
qu = GeneratorEnqueuer(keras_gen(),pickle_safe=True)
qu.start(5,10)
start = time.time()
for i in range(100):
    while not qu.queue.qsize():
        time.sleep(0.5)
    result = qu.queue.get()
print("Took Keras",time.time()-start)
```

    Took Keras 20.02438259124756

Pretty much the same as the executor. But, the data that `GeneratorEnqueuer` will provide won't be in order.


# Conclusion

In this post, we've shown how to use an executor. In the future, this work should be integrated with `Keras` `model.*_generator`.

**UPDATE** This feature has been merged into `Keras` 2.0.5 and it is named Sequence. 
