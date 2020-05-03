from libsvm import svmutil
import numpy as np

#Very Basic SVM api using LIBSVM:
#Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
#vector machines. ACM Transactions on Intelligent Systems and
#Technology, 2:27:1--27:27, 2011. Software available at
#http://www.csie.ntu.edu.tw/~cjlin/libsvm

class SupportVectorMachine:
    """Fit Classification-Problem using a Support Vector Machine"""

    def __init__(self, kernel_type = 'RBF'):
        self.kernel_type = kernel_type
        self.kernel_dict = {'linear' : 0, 'RBF' : 2}
        self.model = None

    def fit(self, X, y):
        task = svmutil.svm_problem(y, X)
        params = svmutil.svm_parameter('-s 0 -t %s -q'
                                        %self.kernel_dict[self.kernel_type])
        self.model = svmutil.svm_train(task, params)

    def predict(self, X):
        labels, acc, vals = svmutil.svm_predict([], X, self.model, options='-q')
        return np.array(labels)
