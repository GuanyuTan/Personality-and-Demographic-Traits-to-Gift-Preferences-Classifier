from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

def confusion_matrix(y_true,y_pred):
  confusion_matrix_ = np.sum(multilabel_confusion_matrix(y_true, y_pred),axis=0)
  recall = confusion_matrix_[1,1]/(confusion_matrix_[1,1]+confusion_matrix_[1,0])
  print("Confusion Matrix\n", confusion_matrix_)
  print(classification_report(y_true,y_pred))
