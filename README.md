# NumpyCL
## What is it?
NumpyCL is a Python3 library built on top of OpenCL for fast computation

## Why it exists?
To address a problem I had and still have while learning Deep Neural Networks.
Writing fast NeuralNets from scratch without TensowFlow or Pytorch.
Fast could also mean C, but as we have Windows and Linux and MAC and I am not a startup, I do not have the time.
And also adding a small brick to the NVIDIA's CUDA monopoly.

## But...WHY??
Most of the tutorials out there teach Neural Nets directly in TensorFlow in 10 lines of code.
Problem with this is that nobody has any faint idea of what is happening.
I like to go about everything I learn the Feynman Way:
  `What I cannot create, I do not undestand`
  or
  `Know how to solve every problem that has been solved`
So by reading a blog post with some theory and then writing those 2 pages of math in a single line of code is not really learning. If someone asks me why this works, I will have to give them a link and in one week time, I won't remember either how to reproduce that result.

However,
If one should stay and implement that math from scratch, I guarantee he will never forget it.

## What works
Currently only addition and substraction of 2 vectors with:
  - identical length
  - identical type
  - only one type supported(numpy.uint8)
I am developing this as I go through the Standford's CS213n course on ComputerVision.
Once I have more time, I will refactor code and add useful commentaries.

## Tests
No tests are developed... YET!

## Early Future improvements
Broadcasting will be added(the same as numpy)
Matrix Multiplication

