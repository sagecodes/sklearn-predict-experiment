from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# create simple training data set
# [height, weight, shoe size]
train_X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

train_Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

tree = tree.DecisionTreeClassifier()
bayes = GaussianNB()

# Train your data with Decision Tree 
# http://scikit-learn.org/stable/modules/tree.html
tree = tree.fit(train_X, train_Y)

# Train your data with Bayes
# http://scikit-learn.org/stable/modules/naive_bayes.html
bayes = bayes.fit(train_X, train_Y)

test_X = [[150, 40, 30], [176, 69, 43], [188,92,48],[184,84,44],[183,83,44],
		  [166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
test_Y = ['female', 'male', 'male','male','male','female','female',
	      'female','male','male']



tree_prediction = tree.predict(test_X)
bayes_prediction = bayes.predict(test_X)


print("Decision Tree: ",accuracy_score(test_Y,tree_prediction))
print("Naive Bays: " ,accuracy_score(test_Y,bayes_prediction))
