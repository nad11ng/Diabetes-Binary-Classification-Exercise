import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from ydata_profiling import ProfileReport
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("diabetes.csv")
# print(data)
# print(data.info)
# print(data.describe())
stats = data.describe()

# profile = ProfileReport(data, title="Diabetes Profiling Report")
# profile.to_file("diabetes_report.html")

x = data.drop(labels="Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=100)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25)
#First attempt only use Train Set and Test Set

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 3, 5, 7],
}
# model = RandomForestClassifier(random_state=200)
# model= LogisticRegression()
# model = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=200),
#     param_grid=params,
#     scoring="recall",
#     cv=6,
#     verbose=1,
#     n_jobs=4
# )
# model.fit(x_train, y_train)
#
#
# y_predict = model.predict(x_test)
# print(model.best_score_)
# print(model.best_params_)
#
#
# # for i, j in zip(y_predict, y_test):
# #     print("Prediction: {}. Actual value: {}".format(i, j))
# # print("Accuracy: ", accuracy_score(y_test, y_predict))
# # print("Precision: ", precision_score(y_test, y_predict))
# # print("Recall: ", recall_score(y_test, y_predict))
# # print("F1: ", f1_score(y_test, y_predict))
#
# print(classification_report(y_test, y_predict))
# print(confusion_matrix(y_test, y_predict))

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
