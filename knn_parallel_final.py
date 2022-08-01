import data_prep as dp
import numpy as np
import time
from collections import defaultdict

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# convert all the dataset into ndarrays
# all of the data in these list is in float64, here cast X_train_list to float32
# is out of parallelization concern
X_train_list = dp.X_train.to_numpy(dtype ='float32')
X_test_list = dp.X_test.to_numpy(dtype ='float32')
y_train_list = dp.y_train.to_numpy()
y_test_list = dp.y_test.to_numpy()

attribute_number = X_train_list.shape[1]

def parallel_train(X_train_list, test):
    '''
        parameters: 
        X_train_list: a 2D ndarray containing all the training points
        test: a specific test point represented as an ndarray

        return: dest
        dest in an 1D array storing distance pairs where tid is corresponed to index
    '''

    mod = SourceModule("""
        __global__ void parallelize(float *distances, 
                    float *X_train_list, float *test, int attribute_number, int size)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int squared_dist = 0;
            for(int attr_idx = 0; attr_idx < attribute_number; ++attr_idx)
            {
                if(tid * attribute_number + attr_idx < size) {
                    squared_dist += (test[attr_idx] - X_train_list[tid * attribute_number + attr_idx])*
                    (test[attr_idx] - X_train_list[tid * attribute_number + attr_idx]);
                }
            }
            
            float dist = sqrt(float(squared_dist));

            // put the value into the array, where tid is the index
            distances[tid] = dist;  
        }
    """ 
    )

    parallelize = mod.get_function("parallelize")

    # distances is a 1D ndarray that will hold the distances from each training point to this test point
    distances = np.zeros(len(X_train_list), dtype=np.float32)

    # make sure it's row-major and in float32
    X_train_list = X_train_list.copy(order='C')
    X_train_list = X_train_list.astype(np.float32)

    # remember to change the type of this ndarray even if I have done it
    test = test.astype(np.float32)

    # flatten the ndarray to fit the CUDA kernel structure
    X_train_list = X_train_list.flatten(order='C')

    # Now the size would be flattened size...
    size = len(X_train_list)

    # each thread is reponsible for one training point
    threads_per_block = 1024

    block_count = int((size + threads_per_block - 1)/threads_per_block)

    # 1D grid, 1D blocks
    parallelize(
        cuda.Out(distances), cuda.In(X_train_list), cuda.In(test), np.int32(attribute_number), np.int32(size),
        block=(1024,1,1), grid=(block_count,1,1)
    )

    # an ndarray constains the euclidean distances from all the train points
    return distances



class kNN():

    def __init__(self, k):
        self.k = k

   
    def predict(self, X_test_list):
        '''
        compute the distance, then do the prediction
        '''
        # calculate the distances
        predictions = []
        # for every test point in 9767 of test_list
        for test in X_test_list:
            # every test point has ONE distances array
            distances = parallel_train(X_train_list, test)

            # now we have the distances list from this specific test point to each train point in inserted order
            sorted_index = np.argsort(distances)
            # contains the ndarray contains the top k least-distanced indices
            top_k_index = sorted_index[:self.k]
            predictions.append(self.vote(top_k_index))
        return predictions
    
    def vote(self, indices):
        '''
            vote:
            get the labels of top k from y_train_list, and vote for the most freq occurence

            parameter(s):
            a numpy.ndarray

            return:
            the class for this test point
        '''
        train_labels = []
        for dist_idx in indices:
            train_labels.append(y_train_list[dist_idx])

        label_counter = defaultdict(int)
        for l in train_labels:
            label_counter[l] += 1
        
        max_label = max(label_counter, key=label_counter.get)
        return max_label


    def get_accuracy(self, y_pred, y_test_list):
        '''
            calc accuracy:
            check the similarity between the prediction list and y_test_list
        '''
        accuracy = 0
        for l in range(len(y_test_list)):
            if y_pred[l] == y_test_list[l]:
                accuracy += 1
        
        accuracy /= len(y_test_list)
        return accuracy

start_time = time.time()
clf = kNN(k=24)
predictions = clf.predict(X_test_list)
accuracy = clf.get_accuracy(predictions, y_test_list)
end_time = time.time()
print('Test Accuracy : {:.3}'.format(accuracy))
print ("Time elapsed:", end_time - start_time)