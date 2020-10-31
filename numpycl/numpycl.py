#
# Vadd
#
# Element wise addition of two vectors (c = a + b)
# Asks the user to select a device at runtime
#

import pyopencl as cl
import numpy as np

import deviceinfo
from time import time

# Tolerance used in floating point comparisons
#
# Kernel: vadd
# 
# To computer the elementwise sum c = a + b
#
# Input: a and b float vectors of length count
# Output: c float vector of len count holding the sum a + b
#
kernelsource = """
__kernel void vadd(
  __global uchar* a,
  __global uchar* b,
  __global uchar* c,
  const unsigned int count)
{
  int i = get_global_id(0);
  if (i < count)
    c[i] = a[i] + b[i];
}
__kernel void vsub(
  __global uchar* a,
  __global uchar* b,
  __global uchar* c,
  const unsigned int count)
{
  int i = get_global_id(0);
  if (i < count) {
    if (a[i] < b[i])
      c[i] = b[i] - a[i];
    else
      c[i] = a[i] - b[i];
  }
}
"""

class NumpyCL(object):
  def __init__(self):
    # Main procedure
    # Asks the user to select a platform/device on the CLI
    self.context = cl.create_some_context()

    # Print device info
    # deviceinfo.output_device_info(self.context.devices[0])

    # Create a command queue
    self.queue = cl.CommandQueue(self.context)

    # Create the compute program from the source bufer
    self.program = cl.Program(self.context, kernelsource).build()


  def prepare_d_rsc(self, h_array1, h_array2, h_res): 
    # Create buffers in device memory to hold data and copy it from the Host
    # d_ prefix is for device resources
    mf = cl.mem_flags
    d_a = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_array1)
    d_b = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_array2)
    # Create output c array in device memory
    d_c = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_res.nbytes)
    return d_a, d_b, d_c


  def sum(self, h_array1, h_array2):
    """
    Gets 2 numpy arrays as input
    Returns a third numpy array = elements-wise sum of the first 2 array
    """
    # empty c vector to hold the sum
    count = len(h_array1)
    h_res = np.empty(count).astype(np.uint8)
    # TODO: check to be the same size
    d_a, d_b, d_c = self.prepare_d_rsc(h_array1, h_array2, h_res)

    # Start the timer
    rtime = time()
    # Execute the kernel over the entire range of our 1d input
    # allow OpenCL runtime to select the workgroup items for the device
    vadd = self.program.vadd
    # Each function arg has an entry
    # If entry is not a scalar, value is None
    vadd.set_scalar_arg_dtypes([None, None, None, np.uint32])
    vadd(self.queue, h_array1.shape, None, d_a, d_b, d_c, count)

    # Wait for the commands to finish before reading back
    self.queue.finish()
    rtime = time() - rtime
    print("The kernel ran in", rtime, "seconds")
    # Read back the results from the compute device
    cl.enqueue_copy(self.queue, h_res, d_c)
    return h_res


  def sub(self, h_array1, h_array2):
    # empty c vector to hold the sum
    count = len(h_array1)
    h_res = np.empty(count).astype(np.uint8)
    # TODO: check to be the same size
    d_a, d_b, d_c = self.prepare_d_rsc(h_array1, h_array2, h_res)

    # Start the timer
    rtime = time()
    # Execute the kernel over the entire range of our 1d input
    # allow OpenCL runtime to select the workgroup items for the device
    vsub = self.program.vsub
    vsub.set_scalar_arg_dtypes([None, None, None, np.uint32])
    vsub(self.queue, h_array1.shape, None, d_a, d_b, d_c, count)
    self.queue.finish()
    rtime = time() - rtime
    print("The kernel ran in", rtime, "seconds")
    # Read back the results from the compute device
    cl.enqueue_copy(self.queue, h_res, d_c)
    return h_res

