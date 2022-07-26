import data_prep as dp
from sklearn.neighbors import KNeighborsClassifier

# k is expected to be 24(0.805) or 33(0.806)
clf = KNeighborsClassifier(n_neighbors=24)

clf.fit(dp.X_train, dp.y_train)

print("Test set predictions: {}".format(clf.predict(dp.X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(dp.X_test, dp.y_test)))