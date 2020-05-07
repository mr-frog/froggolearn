from libsvm import svmutil
import numpy as np

#Very Basic SVM api using LIBSVM:
#Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
#vector machines. ACM Transactions on Intelligent Systems and
#Technology, 2:27:1--27:27, 2011. Software available at
#http://www.csie.ntu.edu.tw/~cjlin/libsvm

class SupportVectorMachine:
    """Fit Classification-Problem using a Support Vector Machine"""

    def __init__(self, svm_type = 'CSVC', kernel_type = 'RBF', verbose=False):
        self.svm_type = svm_type
        self.svm_dict = {'CSVC' : 0}
        self.kernel_type = kernel_type
        self.kernel_dict = {'linear' : 0, 'RBF' : 2}
        if verbose == True:
            self.quiet = ''
        else:
            self.quiet = '-q'
        self.model = None

    def fit(self, X, y):
        if self.quiet == '':
            print('[LIBSVM]:\n')
        task = svmutil.svm_problem(y, X)
        params = svmutil.svm_parameter('-s %s -t %s %s'
                                        %(self.svm_dict[self.svm_type],
                                          self.kernel_dict[self.kernel_type],
                                          self.quiet))
        self.model = svmutil.svm_train(task, params)

    def predict(self, X):
        labels, acc, vals = svmutil.svm_predict([], X, self.model, options='-q')
        return np.array(labels)
