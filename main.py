from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import data_clean
import Visualize_tree as Vstree

path = 'D:\ebm\MLDE PMA\default_data.csv'
data, target = data_clean.clean(path)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

maxdep = []
minsample = []
maxfeature = []
for i in range(15, data.shape[1], 10):
    minsample.append(i * 30)
    maxdep.append(i * 3)
    maxfeature.append(i)
tuned_parameters = [{'criterion': ['gini'],
                     'max_depth': maxdep,
                     'min_samples_split': minsample,
                     'max_features': [data.shape[1]]}]

scores = ['recall']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DTC(), tuned_parameters, cv=3,
                       scoring=score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")

tree_model = DTC(criterion='gini', max_depth=10, max_features=53, min_samples_split=1000)
tree_model_fit = tree_model.fit(X_train, y_train)
predicted = tree_model_fit.predict(X_test)
print(precision_score(y_test, predicted))
print(accuracy_score(y_test, predicted))
print(recall_score(y_test, predicted))
print(f1_score(y_test, predicted))

Vstree.visualizeTree(tree_model, data.columns.values, targetname=['not', 'default'])

