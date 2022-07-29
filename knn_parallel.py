import data_prep as dp
import numpy as np
import time
from math import sqrt
from collections import defaultdict

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# convert all the dataset into ndarrays
X_train_list = dp.X_train.to_numpy()
X_test_list = dp.X_test.to_numpy()
y_train_list = dp.y_train.to_numpy()
y_test_list = dp.y_test.to_numpy()

attribute_number = X_train_list.shape[1]

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


    # initialize an ndarray as the output dsetination
    distance = np.empty(attribute_number, dtype=np.float32)
    
    calculate_distance(
        cuda.Out(distance), cuda.In(train), cuda.In(test),
        block=(7,2,1), grid=(1,1,1)
    )
    
    # do sum and sqrt on the output vector(ndarray)
    dist = sqrt(np.sum(distance))
    return dist

class kNN():

    def __init__(self, k):
        self.k = k

    '''
        compute the distance, then do the prediction
    '''
    def predict(self, X_test_list):
        # calculate the distances
        predictions = []

        # for every test point in 9767 of test_list
        for test in X_test_list:
            # a distances array for every test point
            distances = []


            # for every train point in the train list of about 39073
            for train in X_train_list:
                dist = euclidean_distances(train, test)
                distances.append(dist)


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