from lightgbm import LGBMClassifier
from sklearn.ensemble import (GradientBoostingClassifier, 
                              RandomForestClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt


class TreeLogit:
    def __init__(
        self,
        tree_method='gbdt',
        n_estimators=100,
        max_depth=5,
        ):
        
        self.tree_method = tree_method
        self._support_trees = ['lgb', 'rf', 'gbdt']
        if tree_method == 'gbdt':
            self.gdr_tree = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth)
        elif tree_method == 'rf':
            self.gdr_tree = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth)
        elif tree_method == 'lgb':
            self.gdr_tree = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        
        self.gdr_enc = OneHotEncoder()
        self.gdr_logit = LogisticRegression()
        
    def fit(self, X, y, train_logit_ratio=0.2):
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X, y, test_size=train_logit_ratio)
        tree = self.gdr_tree
        enc = self.gdr_enc
        logit = self.gdr_logit
        
        # fit model
        tree.fit(X_train, y_train)
        if self.tree_method == 'gbdt':
            train_cls, train_cls_lr = tree.apply(X_train)[:, :, 0], tree.apply(X_train_lr)[:, :, 0]
        elif self.tree_method == 'rf':
            train_cls, train_cls_lr = tree.apply(X_train), tree.apply(X_train_lr)
        elif self.tree_method == 'lgb':
            train_cls, train_cls_lr = tree.predict(X_train, pred_leaf=True), tree.predict(X_train_lr, pred_leaf=True)
        
        enc.fit(train_cls)
        lg_input = enc.transform(train_cls_lr)
        logit.fit(lg_input, y_train_lr)
        
        self.gdr_tree = tree
        self.gdr_enc = enc
        self.gdr_logit = logit
        
    def predict(self, X):
        tree = self.gdr_tree
        enc = self.gdr_enc
        logit = self.gdr_logit
        
        if self.tree_method == 'gbdt':
            test_cls = tree.apply(X)[:, :, 0]
        elif self.tree_method == 'rf':
            test_cls = tree.apply(X)
        elif self.tree_method == 'lgb':
            test_cls = tree.predict(X, pred_leaf=True)
            
        lg_input = enc.transform(test_cls)
        res = logit.predict_proba(lg_input)[:, 1]
        return res
    
    @staticmethod
    def evaluate(y, pred):
        auc = roc_auc_score(y, pred)
        acc = accuracy_score(y, pred.round())
        print("AUC: {} || ACC: {}".format(auc, acc))
        
    @staticmethod
    def tree_exp(X_train, y_train, X_test, tree_method='lgb'):
        if tree_method == 'gbdt':
            tree = GradientBoostingClassifier()
        elif tree_method == 'rf':
            tree = RandomForestClassifier()
        elif tree_method == 'lgb':
            tree = LGBMClassifier()
        
        enc = OneHotEncoder()
        logit = LogisticRegression()
        
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.2)
        tree.fit(X_train, y_train)
        
        if tree_method == 'gbdt':
            train_cls = tree.apply(X_train)[:, :, 0]
            train_lr_cls = tree.apply(X_train_lr)[:, :, 0]
            test_cls = tree.apply(X_test)[:, :, 0]
        elif tree_method == 'rf':
            train_cls = tree.apply(X_train)
            train_lr_cls = tree.apply(X_train_lr)
            test_cls = tree.apply(X_test)
        elif tree_method == 'lgb':
            train_cls = tree.predict(X_train, pred_leaf=True)
            train_lr_cls = tree.predict(X_train_lr, pred_leaf=True)
            test_cls = tree.predict(X_test, pred_leaf=True)
            
        enc.fit(train_cls)
        logit.fit(enc.transform(train_lr_cls), y_train_lr)
        
        res = tree.predict_proba(X_test)[:, 1]
        lr_res = logit.predict_proba(enc.transform(test_cls))[:, 1]
        
        return res, lr_res
    
    @staticmethod
    def plot_exp(curve_dict):
        plt.figure(figsize=(9, 5))
        plt.plot([0, 1], [0, 1], 'k--')
        for key, dct in curve_dict.items():
            tree_res = dct['tree']
            tree_lr_res = dct['tree_lr']
            plt.plot(*tree_res, label=f'{key}', linestyle="--")
            plt.plot(*tree_lr_res, label=f'{key}_logit')
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.title('ROC', fontsize=14)
        plt.tick_params(labelsize=13)  # 设置刻度字体大小
        plt.grid()
        plt.legend(loc=4)
        plt.show()
    
    def make_experiments(self, X_train, y_train, X_test, y_test):
        reses = []
        _trees = self._support_trees
        for tree_method in _trees:
            tree_res, tree_lr_res =\
                self.tree_exp(X_train, y_train, X_test, tree_method=tree_method)
            reses.append((tree_res, tree_lr_res))
            tree_auc = roc_auc_score(y_test, tree_res)
            tree_lr_auc = roc_auc_score(y_test, tree_lr_res)
            tree_acc = accuracy_score(y_test, tree_res.round())
            tree_lr_acc = accuracy_score(y_test, tree_lr_res.round())
            print('-' * 50)
            print('{} || AUC: {:.4f} || ACC: {:.4f} .'.format(tree_method, tree_auc, tree_acc))
            print('{} + LR || AUC: {:.4f} || ACC: {:.4f} .'.format(tree_method, tree_lr_auc, tree_lr_acc))
            
        res_dict = dict(zip(_trees, reses))
        curves = dict()
        for key, (tree_pred, tree_lr_pred) in res_dict.items():
            fpr_tr, tpr_tr, _ = roc_curve(y_test, tree_pred)
            fpr_lr, tpr_lr, _ = roc_curve(y_test, tree_lr_pred)
            curves[key] = {'tree': (fpr_tr, tpr_tr), 'tree_lr': (fpr_lr, tpr_lr)}
            
        self.plot_exp(curves)
            
    
        
        
        