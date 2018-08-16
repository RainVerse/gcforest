from gcForest_gdbt import gcForest_gdbt
from data_preprocess import load_oversampled_data
import time
from sklearn.externals import joblib

X_train, y_train = load_oversampled_data(frac=0.1)
print('datasets size:' + str(len(X_train)))

t0 = time.time()
gcf = gcForest_gdbt(shape_1X=30, window=8, tolerance=0.0)
gcf.fit(X_train, y_train)
t1 = time.time()
print("training took {} s".format(t1 - t0))
joblib.dump(gcf, 'model/'+str(len(X_train))+'_1_gdbt.sav')


from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc,accuracy_score
from data_preprocess import load_original_data

X_test, y_test = load_original_data()

pred_X, pred_proba = gcf.predict(X_test)
y_proba = pred_proba[:, 1]

accuracy = accuracy_score(y_true=y_test, y_pred=pred_X)
print('gcForest accuracy : {}'.format(accuracy))

precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_proba)
average_precision = average_precision_score(y_test, y_proba)
print(average_precision)


fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_proba)
roc_auc = auc(fpr, tpr)
print(roc_auc)
