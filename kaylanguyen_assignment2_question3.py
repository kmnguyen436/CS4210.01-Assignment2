#-------------------------------------------------------------------------
# AUTHOR: Kayla Nguyen
# FILENAME: kaylanguyen_assignment2_question3
# SPECIFICATION: Finding the LOO_CV error rate for KNN = 1
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
          db.append (row)

category_dict = {
        "+": 1,
        "-": 2
}
#loop your data to allow each instance to be your test set
nCorrect = 0
nWrong = 0
for i, instance in enumerate(db):

    #add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    #transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages

    #--> add your Python code here
    X = [[int(db[row][col]) for col in range(len(db[row])-1)] for row in range(len(db))]
    Y = [category_dict[db[row][col]] for row in range(len(db)) for col in range(len(db[row]) - 1, len(db[row]))]
    del X[i]
    del Y[i]
    testSample = [[int(instance[0]), int(instance[1])], category_dict[instance[2]]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample[0]])[0]


    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted == testSample[1]:
        nCorrect = nCorrect + 1
    else:
        nWrong = nWrong + 1

errorRate = nWrong/(nCorrect+nWrong)
#print the error rate
#--> add your Python code here
print('Error rate: ' + str(errorRate))