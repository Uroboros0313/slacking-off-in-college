# GBDT+LR

## 原理:
1. 集成树模型:
    - Bagging: RandomForest
    - Boosting: GradientBoostingTree, XGBoost, LightGBM

集成树模型有多棵树, 假设每棵树是完全树且`max_depth=3`, 那么每棵子树有8个`leaf`, 假设一共100棵树(即`n_estimator=100`), 每个样本在每棵树落到的位置可以用一个8位01编码表示, 则一共可以得到100个8位数01编码, 就是GBDT输出的特征。