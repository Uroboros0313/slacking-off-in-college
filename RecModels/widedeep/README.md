# Wide&Deep

- wide(记忆): 线性回归, 同时输入原始特征和特征交叉积。本实现中, wide部分输入**交叉积特征和dense特征**

    - 交叉积: 对类别特征计算共现后进行one-hot
$$
\phi_k(x) = \sum^d_{i=1} x_i^{cki}　　cki\in{0, 1}\tag{1}
$$

> 原论文描述: a cross-product transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if the constituent features (“gender=female” and “language=en”) are all 1, and 0 otherwise.




- deep(泛化): dnn, 类别特征作为embedding输入。本实现中输入dense特征和sparse特征embedding 