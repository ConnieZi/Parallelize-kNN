
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# Transfer data onto the device
# In PyCuda, we will mostly transfer data from numpy arrays on the host
a = numpy.random.randn(4, 4)
# most nvidia devices only support single precision
a = a.astype(numpy.float32)

# somewhere to transfer data to, and allocate memory on the device
a_gpu = cuda.mem_alloc(a.nbytes)
# transfter the data to the GPU
cuda.memcpy_htod(a_gpu, a)


'''
    double each entry in a_gpu
'''
# feed the CUDA C code into the constructor of a pycuda.compiler.SourceModule
mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
""")

func = mod.get_function("doublify")
# specify a_gpu as the argument, and a block size of 4*4
func(a_gpu, block=(4,4,1))

# # shortcuts for explicit memory copies. This will replace a.
# func(cuda.InOut(a), block=(4,4,1))

# fetch the data back from the GPU and display it
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)


# '''
#     Advanced Topics
# '''
# # Structures
# # double a number of variable length arrays

# # is this for one block?
# mod = SourceModule("""
#     struct DoubleOPeration {
#         int datalen, __padding; //so 64-bit ptrs can be aligned
#         float *ptr;
#     };

#     __global__ void double_array(DoubleOperation *a){
#         // each block in the grid will double one of the arrays
#         a = &a[blockIdx.x];
#         for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
#             a->ptr[idx] *= 2;
#         }
#     }

# """
# )

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(5).astype(numpy.float32)
b = numpy.random.randn(5).astype(numpy.float32)

print(a)
print(b)

dest = numpy.zeros_like(a)
multiply_them(
        cuda.Out(dest), cuda.In(a), cuda.In(b),
        block=(5,1,1), grid=(1,1))

print(dest)