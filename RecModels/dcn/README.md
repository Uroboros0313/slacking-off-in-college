# DCNv1(DCN-Vector)

分为CrossNet和DNN部分

## Vector:    

$$x_i = x_i + x_i\odot(W_lx_i + b_l)$$

其中$W_l\in R^{H_c,1}$, 即将$x_i$降维到1维再与原本的$x_i$点积

# DCNV2(DCN-Matrix)

## Matrix:    

$$x_i = x_i + x_i\odot(W_lx_i + b_l)$$

其中$W_l\in R^{H_c,H_c}$, 即将$x_i$不降维再与原本的$x_i$点积

## Mix

- 使用了MOE (Mixuture of experts)方法, 每个expert有一个gate score, 这个分数的参数在$n_{cross}$中共享
- 每层每个expert有一个分解的参数, 参考了奇异值分解的思想将上述的$W_l$分解为$UCV$
$$E(x_i) = x^i\odot(U\cdot g(C\cdot g(V^Tx_i))) + b$$

## 连接方法

- Stacked: 把CrossNet输出的内容输入DNN
- Parallel: 连接两者输入最后的final linear
