from sklearn.metrics import accuracy_score
from plot_functions import plot_pr_curve,plot_roc_curve
from data_preprocess import load_original_data
from sklearn.externals import joblib

X_test, y_test = load_original_data()

gcf = joblib.load('model/56863_1_gdbt.sav')
pred_X, pred_proba = gcf.predict(X_test)
y_proba = pred_proba[:, 1]

accuracy = accuracy_score(y_true=y_test, y_pred=pred_X)
print('gcForest accuracy : {}'.format(accuracy))
auc=plot_roc_curve(y_test,y_proba)
aprs=plot_pr_curve(y_test,y_proba)
print('auc={},aprs={}'.format(auc,aprs))


