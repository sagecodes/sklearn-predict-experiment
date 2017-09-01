from sklearn import tree
from sklearn.naive_bayes import GaussianNB

# create simple data set
# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

tree = tree.DecisionTreeClassifier()
bayes = GaussianNB()

# Train your data with Decision Tree 
# http://scikit-learn.org/stable/modules/tree.html
tree = tree.fit(X, Y)

# Train your data with Bayes
# http://scikit-learn.org/stable/modules/naive_bayes.html
bayes = bayes.fit(X, Y)

tree_prediction = tree.predict([[150, 40, 30]])

bayes_prediction = bayes.predict([[177, 70, 43]])


print(bayes_prediction)
