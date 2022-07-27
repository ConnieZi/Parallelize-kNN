import data_prep as dp
import numpy as np
import time
from math import sqrt
from collections import defaultdict

# convert all the dataset into ndarrays
X_train_list = dp.X_train.to_numpy(dtype ='int64')
X_test_list = dp.X_test.to_numpy(dtype ='int64')
y_train_list = dp.y_train.to_numpy(dtype ='int64')
y_test_list = dp.y_test.to_numpy(dtype ='int64')


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
                squared_dist = 0
                # for each attribute in 14 attributes
                for attr_idx in range(X_train_list.shape[1]):
                    squared_dist += np.square(test[attr_idx]-train[attr_idx])
                dist = sqrt(squared_dist)
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