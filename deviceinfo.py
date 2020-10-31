#
# Devide Info
#
# Function to output key params about the input OpenCL device
#

import pyopencl as cl
from sys import stdout

def output_device_info(device_id):
  stdout.write("Device is ")
  stdout.write(device_id.name)
  if device_id.type == cl.device_type.GPU:
    stdout.write("GPU from ")
  elif device_id.type == cl.device_type.CPU:
    stdout.write("CPU from")
  else:
    stdout.write("non CPU or GPU processor from ")
  stdout.write(device_id.vendor)
  stdout.write(" with a max of ")
  stdout.write(str(device_id.max_compute_units))
  stdout.write(" compute units\n")
  stdout.flush()
