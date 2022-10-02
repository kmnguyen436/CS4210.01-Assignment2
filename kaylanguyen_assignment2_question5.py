#-------------------------------------------------------------------------
# AUTHOR: Kayla Nguyen
# FILENAME: kaylanguyen_assignment2_question5
# SPECIFICATION: Training sets and use the test to output classification of each test instance for confidence of at least 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 43 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
#reading the training data in a csv file
dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append(row)

category_dict = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3,
    "Hot": 1,
    "Mild": 2,
    "Cool": 3,
    "Normal": 1,
    "High": 2,
    "Strong": 1,
    "Weak": 2,
    "Yes": 1,
    "No": 2
}
#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = [[category_dict[dbTraining[row][col]] for col in range(1,len(dbTraining[row]) - 1)] for row in range(len(dbTraining))]

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [category_dict[dbTraining[row][col]] for row in range(len(dbTraining)) for col in range(len(dbTraining[row]) - 1, len(dbTraining[row]))]


#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTesting = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTesting.append(row)
            print(row)
X_t = [[category_dict[dbTesting[row][col]] for col in range(1,len(dbTesting[row]) - 1)] for row in range(len(dbTesting))]

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for instance in dbTesting:
    X_t = [category_dict[instance[col]] for col in range(1, len(instance) - 1)]
    if clf.predict_proba([X_t])[0][0] > 0.75 or clf.predict_proba([X_t])[0][1] >= 0.75:
        print(instance[0].ljust(15), instance[1].ljust(15), instance[2].ljust(15), instance[3].ljust(15), instance[4].ljust(15), str(clf.predict([X_t])[0]).ljust(15), clf.predict_proba([X_t])[0])

