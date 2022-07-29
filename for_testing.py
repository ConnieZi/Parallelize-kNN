import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from math import sqrt


def euclidean_distances(train, test):
    '''
        train: a train vector as an ndarray
        test: a test vector as an ndarray
    '''

    mod = SourceModule("""
        __global__ void distance(float *distance, float *train, float *test)
        {
            int tid = threadIdx.x + blockDim.x * threadIdx.y;
            distance[tid] = (train[tid] - test[tid]) * (train[tid] - test[tid]);
        }
    """ 
    )

    calculate_distance = mod.get_function("distance")
    
    # remember to type-cast the array to be in float32 so that C can calculate
    train = train.astype(np.float32)
    test = test.astype(np.float32)


    # initialize an ndarray as the output destination
    distance = np.empty(14, dtype=np.float32)
    calculate_distance(
        cuda.Out(distance), cuda.In(train), cuda.In(test),
        block=(7,2,1), grid=(1,1,1)
    )
    
    
    # print(f'distance array is {distance}')
    # do sum and sqrt on the output vector(ndarray)
    dist = sqrt(np.sum(distance))
    return dist


train = np.random.randint(4, size=14)
test = np.random.randint(4, size=14)

print(f'test array: {train}')
print(f'test array: {test}')

dist = euclidean_distances(train, test)
print(dist)