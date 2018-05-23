---
layout: post
title:  Deterministic data augmentation - Keras 2.1.7
---

In a previous [post](/full-segmentation), I showed how to use [Keras-Transform](https://github.com/Dref360/keras-transform), a library I created to perform data augmentation on segmentation datasets. After discussion with [Francois Chollet](https://github.com/fchollet), I think that his idea is much better and it's been recently merged on Keras. I was excited to try it out!

The code is available in this [gist](https://gist.github.com/Dref360/d4eafccdd1c9d6e87764c43da50ffb19).

### High-level

Instead of building a pipeline of *Sequences* (like in Keras-Transform), the data augmentation will be done inside one. So we should get something like this :

```python
class MySequence(keras.utils.Sequence):
  def __init__(self):
    ...
  def __getitem__(self, idx):
    X, y = ... # Get the inputs
    for i in len(X):
      rotation = get_random_rotation()
      # Apply the same transformation to X and y
      X[i] = rotate_image(X[i], rotation)
      y[i] = rotate_image(y[i]. rotation)
    return X, y
```


### In Keras 2.1.7

Thanks to [vkk800](https://github.com/vkk800), this API has made it into Keras and will be released soon. Here's a **simple** example.

---
We first import OpenCV and the new deterministic transformations.
```python
import cv2
import numpy as np

from keras.preprocessing.image import apply_affine_transform,
                                      apply_channel_shift
```
---
In order to create a pipeline, we will need a function to apply multiple functions on an object.
Thanks to [ladaghini](https://stackoverflow.com/a/11736719/3784713) for his answer.

```python
import functools
def chain_funcs(func_list, params, obj):
  # Apply iteratively `func_list` to `obj`
  return functools.reduce(lambda o, args: args[0](o, **args[1]),
                          zip(func_list, params), obj)
```
---

We only need to create a Sequence. Since this is an example, this will only return a batch of cats.

```python
from keras.utils import Sequence
class MySequence(Sequence):
  def __init__(self):
    self.path = '~/Images/cat.jpg'
    # data augmentation ranges
    self.rotation_range = 20
    self.tx_range = 10
    self.channel_shift_range = 80
    # All the operations we want to apply
    self.pipeline = [apply_affine_transform, apply_channel_shift]

  def get_random_params(self):
    """Those are the **kwargs needed for `apply_affine_transform`
       and `apply_channel_shift`"""
    return [{'theta': np.random.randint(-self.rotation_range, self.rotation_range),
             'tx': np.random.randint(-self.tx_range, self.tx_range),
             'channel_axis': 2},
            {'intensity': np.random.randint(-self.channel_shift_range,
                                             self.channel_shift_range),
             'channel_axis': 2}]

  def __len__(self):
    # Dummy length
    return 10

  def __getitem__(self, idx):
    X = np.array([cv2.resize(cv2.imread(self.path), (100, 100))
                  for _ in range(10)]).astype(np.float32)  # Fake batch of cats
    y = np.copy(X) # We copy the inputs.
    for i in range(len(X)):
      # We get the random params
      params = self.get_random_params()
      # Apply the same function to X and y
      X[i] = chain_funcs(self.pipeline, params, X[i])
      y[i] = chain_funcs(self.pipeline, params, y[i])
    return X, y
```

### Results

Using this `Sequence`, we can get the following results. (`X` is on the top row and `y` on the second)

![](/images/cat_aug.jpg.png "Results")


Making this work for a real-life application is really easy! We can even have a pipeline for the inputs and one for the targets, this would be useful if the targets are boxes.
The code is available in this [gist](https://gist.github.com/Dref360/d4eafccdd1c9d6e87764c43da50ffb19).

Thank you for reading and I'm always available on Keras' Slack (@Dref360) if you have any question!

Cheers,
Frédéric Branchaud-Charron
