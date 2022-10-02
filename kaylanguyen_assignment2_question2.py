#-------------------------------------------------------------------------
# AUTHOR: Kayla Nguyen
# FILENAME: kaylanguyen_assignment2_question2
# SPECIFICATION: Training sets and testing to find lowest accuracy for each model
# FOR: CS 4210-Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
import row as row
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
trainingSetNum = 1
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)


    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    category_dict = {
        "Young": 1,
        "Prepresbyopic": 2,
        "Presbyopic": 3,
        "Myope": 1,
        "Hypermetrope": 2,
        "Yes": 1,
        "No": 2,
        "Reduced": 1,
        "Normal": 2
    }
    #--> add your Python code here
    # X =
    X = [[category_dict[dbTraining[row][col]] for col in range(len(dbTraining[row]) - 1)] for row in range(len(dbTraining))]
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> addd your Python code here
    # Y =
    Y = [category_dict[dbTraining[row][col]] for row in range(len(dbTraining)) for col in range(len(dbTraining[row]) - 1, len(dbTraining[row]))]
    #loop your training and test tasks 10 times here
    lowestAcc = 100
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        X_t = []
        Y_t = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)


        nCorrect = 0
        nWrong = 0
        for data in dbTest:

            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            X_t = [category_dict[data[col]] for col in range(len(data)-1)]
            Y_t = [category_dict[data[col]] for col in range(len(data) - 1, len(data))]

            class_predicted = clf.predict([X_t])[0]



            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if class_predicted == Y_t:
                nCorrect = nCorrect + 1
            else:
                nWrong = nWrong + 1

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        errorRate = nWrong/(nCorrect+nWrong)
        if (1-errorRate) < lowestAcc:
            lowestAcc = 1-errorRate

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print('final accuracy when training on contact_lens_training_' + str(trainingSetNum) + '.csv: ' + str(lowestAcc))
    trainingSetNum = trainingSetNum + 1


