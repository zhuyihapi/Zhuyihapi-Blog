

---
title: Machine Learning
---

# [Machine Learning](https://www.bilibili.com/video/BV164411b7dx)




## 1 Introduction

### 1.1 Supervised Learn

A "right answer" given

#### Regression

Predict continuous valued output (e.g. housing price)

**Related algorithms**:

- Linear regression
- Neural Networks
- Nearest Neighbor

#### Classification

Discrete valued output (0 or 1)

**Related algorithms**:

- Logistic regression
- K-Nearest Neighbor (KNN)
- Support Vector Machines (SVM)
- Naïve Bayes
- Decision Trees
- Neural Networks



### 1.2 Unsupervised Learn

#### [Cluster](https://zhuanlan.zhihu.com/p/78382376)

**Applications**:

> 聚类：将相似的对象归到同一个簇中，使得同一个簇内的数据对象的相似性尽可能大，同时不在同一个簇中的数据对象的差异性也尽可能地大。

- Market segmentation
- Social network analysis
- Organize computing cluster
- Astronomical data analysis

[**Related algorithms**](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68):

- K-Means Clustering
- Mean-Shift Clustering
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)



<!-- more -->



## 2 Linear Regression with One Variable

### 2.0 Model Representation - Notation

$m$: number of training examples

$x$'s: input variable/feature

$y$'s: output variable/feature

$(x,y)$: a training example

$(x^i,y^i)$: $i$ represents the $i^{th}$



### 2.1 Model and Cost Function

**Hypothesis 假设函数**: 
$$
h_θ(x)=θ_0+θ_1x
$$
**Parameters**: $\theta_0,\theta_1$

> $\theta_1$代表斜率而$\theta_0$则代表由代价函数计算出的差值

[**Cost Function 代价函数**](https://www.cnblogs.com/geaozhang/p/11442343.html):
$$
J(θ_0,θ_1)=\frac {1}{2m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})^2
$$

> minimize squared error cost function (最小化平方差代价函数)

**Goal**: $\min_{\theta_0,\theta_1}J(\theta_0,\theta_1)$



### 2.2 Parameter Learning - Gradient Descent

#### Outline

- Start with some $\theta_0,\theta_1$

- Keep changing $\theta_0,\theta_1$ to reduce $J(\theta_0,\theta_1)$ until we hopefully end up at a minimum

#### Gradient descent algorithm

​	Repeat until convergence {

​		$\theta_j:=\theta_j-\alpha \frac∂{∂\theta_j}J(\theta_0,\theta_1)$	(for j=0 and j=1) 

}

> $:=$ colon equals, which used to denote assignment (赋值运算符)
>
> $\alpha$ is called the learning rate, determined how big a step we take downhill with gradient descent
>
> $\frac∂{∂\theta_j}J(\theta_0,\theta_1)$ is a derivative term (导数项)
>
> **Assert**: simultaneous update $\theta_0,\theta_1$ "at the same time"

#### Gradient descent intuition

$$
\theta_1:=\theta_1-\alpha\frac∂{∂\theta_j}J(\theta_1)
$$

1. $\alpha$

   if the $\alpha$ is too small, gradient descent can be very <u>slow</u>.

   if $\alpha$ is too large, gradient descent can <u>overshoot</u> the minimum. It may fail to converge, or even diverge.

2. Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed.

3. As we approach a local minimum, gradient descent will automatically take smaller step. So, no need to decrease $\alpha$ over time.

#### Gradient descent for linear regression

> Apply gradient descent to minimize squared error cost function 

$$\frac∂{∂\theta_j}J(\theta_0,\theta_1)=\frac∂{∂\theta_j}\frac {1}{2m}\sum_{i=1}^m(\theta_0+\theta_1x^{(i)}-y^{(i)})^2$$		*Expanding the formula*

Substituting 0 and 1 into $j$

$$\theta_0:j=0:\theta=\frac1m\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})$$

$$\theta_1:j=1:\frac∂{∂\theta_1}J(\theta_0,\theta_1)=\frac1m\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})x^{(i)}$$

When work out the derivatives, which is the slope of the cost function *J*, plug them back in to gradient descent algorithm (remember to update simultaneously).

**Convex function**

> It doesn't have any local optimum except for the global optimum

**"Batch" Gradient Descent**

> "Batch": Each step of gradient descent uses all the training examples (entire training set)



## 3 Linear Algebra review (optional)

### 3.1 Matrices and vectors

#### Matrix

> Rectangular array of numbers

3×2 matrix: $\begin{bmatrix}1 & 2 \\ 3 & 4\\ 5&6\end{bmatrix}$	2×3 matrix: $\begin{bmatrix}1 &2&3 \\6& 3 & 4\\ \end{bmatrix}$

**Dimension of matrix**: number of rows $\times$ number of columns

> The above matrix can be also write as $\mathbb R^{3\times2}$ 

**Refer to specific elements of the matrix **(entries of matrix)

$A=\begin{bmatrix}1402&191 \\ 1371 &821\\ 949&1437\\147&1448\end{bmatrix}$

$A_{ij}=$ "$i$,$j$ entry" in the $i^{th}$ row, $j^{th}$ column.

$A_{11}=1402$, $A_{12}=191$, $A_{41}=147$

#### Vector

> An $n\times1$ matrix

$y=\begin{bmatrix}460\\232\\315\\178\end{bmatrix}$

$y_i=i^{th}$ element

> It is often customary to use uppercase letters for matrices and lowercase letters for vectors



### 3.2 Addition and Scalar multiplication



### 3.3 Matrix-vector multiplication

**Details**:

​            $A$           $\times$   $x$       $=$       $y$

$\begin{bmatrix}&&&&&\\ \\ \\ \end{bmatrix}\times\begin{bmatrix}\\ \\ \\ \end{bmatrix} \quad=\quad \begin{bmatrix}\\ \\ \\ \\ \end{bmatrix}$

   m$\times$n matrix        n$\times$1    m-dimensional vector

To get $y_i$, multiply $A$' $i^{th}$ row with elements of vector $x$, and add them up.

**Calculation Tips**：

House sizes: 2104,1216, 1534, 852

Competing hypotheses: $h_\theta(x)=-40+0.25x$

It can be calculated as $\begin{bmatrix}1&2140\\1&1416\\1&1534\\1&852 \end{bmatrix}\times\begin{bmatrix}-40\\0.25\end{bmatrix}$ 



### 3.4 Matrix-matrix multiplication

**Details**:

$A\times B=C$

[m$\times$n]$\times$[n$\times$o]=m$\times$o

The $i^{th}$ column of the matrix $C$ is obtained by multiplying $A$ with the $i^{th}$ column of $B$. (For $i$=1,2,...,0)

**Example**:

$\begin{bmatrix}1&3\\2&5\end{bmatrix}\begin{bmatrix}0&1\\3&2\end{bmatrix}=\begin{bmatrix}1\times0+3\times3&1\times1+3\times2 \\2\times0+5\times3&2\times1+5\times2\end{bmatrix}=\begin{bmatrix}9&7\\15&12\end{bmatrix}$

**Calculation Tips II**：

House sizes: 2104,1216, 1534, 852

Three competing hypotheses: 

1. $h_\theta(x)=-40+0.25x$
2. $h_\theta(x)=200+0.1x$
3. $h_\theta(x)=-150+0.4x$

It can be calculated as $\begin{bmatrix}1&2140\\1&1416\\1&1534\\1&852 \end{bmatrix}\times\begin{bmatrix}-40&200&-150 \\ 0.25&0.1&0.4 \end{bmatrix}$



### 3.5 Matrix multiplication properties

Let $A$ and $B$ are matrices. then is general, $A\times B\ne B\times A$. (**Not commutative**) 

#### Identity Matrix

Denoted $I$ (or $I_{n\times n}$).

Example of identity matrices:

2$\times$2: $\begin{bmatrix}1&0 \\ 0&1\end{bmatrix}$      3$\times$3:$\begin{bmatrix}1&0&0 \\ 0&1&0 \\ 0&0&1\end{bmatrix}$      $\cdots$

For any matrix $A$,

$$
A\cdot I=I\cdot A=A
$$

> Implicit conditions of the formula: 
>
> $A(m\times n)\cdot I(n\times n)=I(m\times m)\cdot A(m\times n)=A(m\times n)$



### 3.6 Inverse and Transpose

> 矩阵的逆和矩阵的转置

Not all numbers have an inverse. (e.g. 0) Likely, not all matrix has an inverse.(e.g.$\begin{bmatrix}0&0 \\ 0&0\end{bmatrix}$)

> Matrices that don't have an inverse are "singular" or "degenerate".

#### Matrix inverse

If A is an m$\times$m matrix, and if it has an inverse,
$$
AA^{-1}=A^{-1}A=I
$$

> An m$\times$m matrix called a square matrix (方阵), only square matrix has an inverse. 

#### Matrix transpose

Example:
$$
A=\begin{bmatrix}1&2&0 \\ 3&5&9\end{bmatrix} \qquad A^T=\begin{bmatrix}1&3 \\ 2&5 \\ 0&9 \end{bmatrix}
$$
Let $A$ be an $m\times n$ matrix, and let $B=A^T$. Then $B$ is an $n\times m$ matrix, and $B_{ij}=A_{ji}$.



## 4 Linear Regression with Multiple Variables

### 4.1 Multiple feature

#### Notation

$n$ = number of features

$x^{(i)}$ = input (features) of $i^{th}$ training example

$x^{(i)}_j$ = value of feature $j$ in $i^{th}$ training example

#### Multivariate linear regression

> 多元线性回归

**Hypothesis**:

$h_θ(x)=θ_0+θ_1x_1+\theta_2x_2+\theta_3x_3+\cdots+\theta_nx_n$

For convenience of notation, define $x_0=1$. Then

$h_θ(x)=θ_0x_0+θ_1x_1+\theta_2x_2+\theta_3x_3+\cdots+\theta_nx_n$

​          $=\theta^Tx$

​          $=\begin{bmatrix}\theta_0&\theta_1&\cdots&\theta_n\end{bmatrix}\begin{bmatrix}x_0 \\ x_1 \\ \cdots \\ x_n\end{bmatrix}$



### 4.2 Gradient descent for multiple variables

- 如何设定假设的参数

- 如何使用梯度下降法来处理多元线性回归

**Hypothesis**: $h_θ(x)=\theta^Tx=θ_0x_1+θ_1x_1+\theta_2x_2+\theta_3x_3+\cdots+\theta_nx_n$

**Parameters**: $\theta_0$,$\theta_1$,...,$\theta_n$

**Cost function**: $J(\theta_0$,$\theta_1$,...,$\theta_n)$$=\frac {1}{2m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})^2$

**Gradient descent**:

​	Repeat {

​		$\theta_j:=\theta_j-\alpha \frac∂{∂\theta_j}J(\theta_0,...,\theta_n)$		

}		(simultaneously update for every $j=0,...,n$)

> $J(\theta_0,...,\theta_n)$ can be instead by $J(\theta)$

**New algorithm** for $n\ge1$:

​	Repeat {

​		$\theta_j:=\theta_j-\alpha\frac {1}{m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})x^{(i)}_j$		

}		(simultaneously update for $\theta_j$ for $j=0,...,n$)

> Previously $(n=1)$ is in 2.2 



### 4.3 Gradient descent in practice I: Feature Scaling

> 梯度下降运算中的实用技巧——特征缩放

#### Feature Scaling

> **Idea**: Make sure flatten are on a similar scale.

E.g. $x_1$= size(0-2000 $feet^2$)

​      $x_2$= number of bedrooms (1-5)

It will be better to limit both $x_1$ and $x_2$ in [0,1]

$x_1=\frac{size(feet^2)}{2000}$

$x_2=\frac{numberOfBedroom}{5}$

> 椭圆相较于正圆需要更多的时间来计算梯度下降

More general, get every feature into approximately a $-1\le x_i\le1$ range.

> 范围的上下限并不是被严格限制的，不过越接近-1和1越好。过大和过小都是不合适的。

#### Mean normalization

Replace $x_i$ with $x_i-µ_i$ to make features have approximately zero mean (Do not apply to $x_0=1$)
**E.g.** 

$x_1=\frac{size(feet^2)-1000}{2000}$

$x_2=\frac{numberOfBedroom-2}{5}$

> $µ_i$ (1000 and 2) is considered as the average value of $x_i$ in training set

$$
x_i\leftarrow \frac{x_i-µ_i}{range(max-min)}
$$



### 4.4 Gradient descent in practice II: Learning rate

- The chapter will center around the learning rate $\alpha$

**Gradient descent**

* $\theta_j:=\theta_j-\alpha \frac∂{∂\theta_j}J(\theta)$
* "Debugging": How to make sure gradient descent is working correctly.
* How to choose learning rate $\alpha$

Declare convergence if $J(\theta)$ decreases by less than $10^{-3}$ in one iteration.

> 这是因为若将$min_\theta J(\theta)$作为纵轴，将迭代次数作为横轴，那么得到的是是一个近似$y=|\frac 1x|$的图像。当迭代次数达到一定量$(\epsilon)$后，梯度下降的量就几乎可以忽略不计了，所以该测试就判断函数已经收敛。不过要选择一个合适的阈值（threshold，$\epsilon$）并不容易。

> 如果你的图像并不是上述的样子，那么说明你的$\alpha$值选取的不恰当。例如图像在0点附近形如指数函数图像，或者呈波浪形，就意味着你的$\alpha$值过大了，函数无法收敛。

**Summary**:

* If $\alpha$ is too small: slow convergence.
* If $\alpha$ is too large: $J(\theta)$ may not decrease on every iteration; may not converge.

Recommended choices for $\alpha$:

..., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,... 



### 4.5 Features and polynomial regression

> 根据特征选择算法以提高效率；使用多项式回归来拟合复杂函数

Housing prices prediction:

$$
h_\theta(x)=\theta_0+\theta_1\times frontage+\theta_2\times depth
$$

> It's better to use $area$ which is equal to $frontage\times depth$ as new feature.

$$
h_\theta(x)=\theta_0+\theta_1\times area
$$

#### Choice of feature

Suppose we have a graph with the price of a house on the vertical axis and the area (size) on the horizontal axis, and we need to choose the function to fit the data recorded on the graph.

$$
h_\theta(x)=\theta_0+\theta_1(size)+\theta_2(size)^2
$$

> 使用二次函数来拟合房价-面积曲线可能在一定范围内是合适的，但是二次函数曲线一定会在达到顶点后下降，房价却不会。所以这并不是一个好的选择。

$$
h_\theta(x)=\theta_0+\theta_1(size)+\theta_2(size)^2+\theta_3(size)^3
$$

> 也许是可行的，但要注意应用特征缩放，从而使三个特征值$(size,(size)^2,(size)^3)$都在大致相同的范围内

$$
h_\theta(x)=\theta_0+\theta_1(size)+\theta_2\sqrt{(size)}
$$

> 上升的曲线，斜率随着x（area）变大逐渐变小，增长趋于平缓，似乎也不错

**Summary**:

You have a choice in what features to use to fix more complex functions to your data!



### 4.6 Normal equation (unfinished)

> 对于某些线性回归问题，求取参数$\theta$最优值的方法。不同与以往的迭代算法（梯度下降的多次迭代来收敛到全局最小值），正规方程提供了一种解析解法，一次性求解$\theta$的最优值。

**Normal equation**: Method to solve for $\theta$ analytically.



#### Compare to Gradient Descent

$m$ training example, $n$ features.

| Gradient Descent            | Normal Equation               |
| --------------------------- | ----------------------------- |
| Need to choice a $\alpha$   | No need to choice a $\alpha$  |
| Needs many iterations       | Don't need to iterate         |
| Work well even $n$ is large | Need to compute $(X^TX)^{-1}$ |
|                             | Slow if $n$ is very large     |


$$
\theta=(X^TX)^{-1}X^Ty
$$
$(X^TX)^{-1}$ is inverse of matrix $(X^TX)$

**Octave**: `pinv(X'*X)*X'*y` 

> `X'` is the transpose of $X$
>
> `pinv` is a function  to compute the inverse of a matrix



<!--unfinished-->

### 4.7 Normal equation and non-invertibility (optional) (unfinished)

<!--unfinished-->



## 5 Octave Tutorial (ignored)

<!--unfinished-->



## 6 Logistic Regression

> 回归算法

### 6.1 Classification

**Classification**

$y\in \{0,1\} $

0: "Negative Class" (e.g., benign tumor)

1: "Positive Class" (e.g., malignant tumor)

> There are multi-class problems as well that y can take value from 0, 1, 2, 3,...

Learning regression isn't fit the classification problem

In the [video](https://www.bilibili.com/video/BV164411b7dx?p=32&t=160) there is an example to explain it. 

Another example:

Classification: y = 0 or 1

​	$H_\theta(x)$ can be $>1$ or $<0$ if we use the linear regression

> Obviously, the label is either 0 or 1.

Logistic Regression: $0\le h_\theta(x)\le1$

> This is a classification algorithm whose output always between 1 and 0. Besides, it's a classification algorithm instead of linear regression algorithm though there is a "regression" in its name.



### 6.2 Hypothesis Representation

> 假设陈述

- What is the function we're going to use to representation hypothesis when we have a classification problem.

#### Logistic Regression Model

​	Want $0\le h_\theta(x)\le1$

$$
h_\theta(x)=g(\theta^Tx)
$$
**Sigmoid function (Logistic function)**: 
$$
g(z)=\frac1{1+e^{-z}}
$$
<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524114255.png" alt="image-20210523152323634" style="zoom:67%;" />

> It's graph is likely function $y=\frac12\tan^{-1}x+\frac12$, it has two asymptote at 0 and 1. And, $h_\theta(0)=0.5$

Thus,
$$
h_\theta(x)=\frac1{1+e^{-\theta^T x}}
$$


> $\theta^Tx\ge0$ then  $h_\theta(x)=1$, $\theta^Tx<0$ then  $h_\theta(x)=0$

**Interpretation of Hypothesis Output**

$h_\theta(x)=$ estimated probability that y = 1 on input x

Example: if $x=\begin{bmatrix}x_0 \\x_1 \end{bmatrix}=\begin{bmatrix}1 \\ tumorSize\end{bmatrix}$

​				$h_\theta(x)=0.7$

Tell patient that 70% chance of tumor being malignant.

#### Mathematical formula definition of the hypothesis for logistic regression 

"Probability that y=1, given x, parameterized by $\theta$": 
$$
P(y=0|x;\theta)+P(y=1|x;\theta)=1\\
P(y=0|x;\theta)=1-P(y=1|x;\theta)
$$


### 6.3 Decision boundary

> 决策界限

- What logistic regression hypothesis function is computing?

  

According to Logistic regression, 

suppose predict "$y=1$" If $h_\theta(x)\ge0.5$

predict "$y=0$" If $h_\theta(x)\le0.5$

#### Decision Boundary

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524114302.png" alt="image-20210523184323276" style="zoom:67%;" />

Suppose the variable procedure to be specified. 

$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2)$

And, $\theta=\begin{bmatrix}-3 \\ 1\\ 1\end{bmatrix}$

Predict "$y=1$"if $-3+x_1+x_2\ge0$

​			         				$x_1+x_2\ge3$

The magenta line is called **Decision Boundary**.

> The decision boundary line is the property of the hypothesis and of the parameters, and not a property of a data set.

#### Non-linear decision boundaries

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524160449.png" alt="image-20210524160448985" style="zoom:50%;" />

Assuming hypothesis likes this: 

$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_2^2)$

And assuming chosen parameters as $\theta=\begin{bmatrix}-1 \\ 0 \\ 0 \\ 1\\ 1\end{bmatrix}$

Then, predict "$y=1$" if $-1+x_1^2+x_2^2\ge0$

​                                               $x_1^2+x_2^2\ge1$

> The training set used to fit the parameters $\theta$



### 6.4 Cost function

- How to automatically choose the parameters $\theta$ to a training set.

- Define the optimization objective or the cost function that used to fit the parameters.



Here is to supervised learning problem of fitting a logistic regression model.

Training set: $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$

$m$ examples $\qquad x\in\begin{bmatrix}x_0 \\ x_1 \\ \cdots \\ x_n\end{bmatrix} \qquad x_0=1,y\in\{0,1\}$

$h_\theta(x)=\frac1{1+e^{-\theta^T x}}$

How to choose parameters $\theta$ ? (The next sections will focus on this problem)



#### Cost function - Logistic regression cost function

​	Linear regression: $J(\theta) =\frac {1}{m}\sum_{i=1}^m\frac12(h_θ(x^{(i)})-y^{(i)})^2$

​	$Cost(h_\theta(x),y)=\frac12(h_\theta(x),y)^2$

> 我们先尝试直接将线性回归函数转化为逻辑回归函数，事实上后者将会是一个参数为$\theta$的非凸函数（non-convex function），这是因为这一部分（$\frac1{1+e^{-\theta^T x}}$）是很复杂的非线性函数。

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524170532.png" alt="image-20210524170532375" style="zoom:80%;" />



**Logistic regression cost function**
$$
Cost(h_\theta(x),y)=
\begin{cases}
-\log(h_\theta(x)) \quad & if \;y=1  \\[1ex]
-\log(1-h_\theta(x))\quad & if \;y=0
\end{cases}
$$


If y = 1

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524171531.png" alt="image-20210524171531895" style="zoom: 80%;" />

If y = 0

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524171740.png" alt="image-20210524171740822" style="zoom:80%;" />



### 6.5 simplified cost function and gradient descent

- Figure out a simpler way to write the cost function

- Also figure out how to apply gradient descent to fit the parameters of logistic regression

#### Logistic regression cost function

$$
J(\theta) =\frac {1}{m}\sum_{i=1}^mCost(h_θ(x^{(i)})-y^{(i)})
$$

$$
Cost(h_\theta(x),y)=
\begin{cases}
-\log(h_\theta(x)) \quad & if \;y=1  \\[1ex]
-\log(1-h_\theta(x))\quad & if \;y=0
\end{cases} \\ \;\\
Note:y=0\;or\;1\;always
$$

Compress them into one equation:

$$
Cost(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$


**Logistic regression cost function**
$$
J(\theta) =\frac {1}{m}\sum_{i=1}^mCost(h_θ(x^{(i)})-y^{(i)})\\
\qquad\qquad\qquad\qquad\qquad\qquad\;\;\; 
=-\frac1m[\sum^m_{i=1}y^{(i)}\log{h_\theta(x^{(i)})} +(1-y^{(i)})\log (1-h_\theta(x^{(i)}))]
$$

> 为什么用这个函数作为逻辑回归的代价函数：这个式子是从统计学中的极大似然法（the principle maximum likelihood estimation）得来的，它是统计学中为不同模型快速寻找参数的方法。并且它还拥有一个良好的性质：它是凸函数。

To fit parameters $\theta$: 
$$
\min_\theta J(\theta)
$$

> find the $\theta$ which minimizes $J(\theta)$

To make prediction given new $x$:

​	Output $h_\theta(x)=\frac1{1+e^{-\theta^T x}}$

> So how to minimize $J(\theta)$, or how to choose parameter $ \theta$ ?



#### Implementation of logistic regression

**Gradient Descent**

$J(\theta)=-\frac1m[\sum^m_{i=1}y^{(i)}\log{h_\theta(x^{(i)})} +(1-y^{(i)})\log (1-h_\theta(x^{(i)}))]$

What $\min_\theta J(\theta)$:	

​	Repeat {

​				$\theta_j:=\theta_j-\alpha\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})x^{(i)}_j$		

​		}		(simultaneously update for $\theta_j$ for $j=0,...,n$)

> Algorithm looks identical to linear regression! But pay attention to $h_\theta(x)$. In linear regression, $h_\theta(x)=\theta^Tx$, and in logistic regression, $h_\theta(x)=\frac1{1+e^{-\theta^T x}}$. The definition of hypothesis has changed, thus actually they are two different things.



### 6.6 Advanced optimization

- Some advanced optimization algorithms 
- Some advanced optimization concepts

> 大大提高逻辑回归的计算速度。:see_no_evil:

#### Optimization algorithm

Cost function $J(\theta)$. Want $\min_\theta J(\theta)$.

Given $\theta$, we have code that compute

- $J(\theta)$
- $\frac∂{∂\theta_j}J(\theta)$      (For $j=0,1,…,n$)

Optimization algorithms:

- Gradient descent
- Conjugate gradient
- BFGS (共轭梯度法)
- L-BFGS

Others algorithm compare to gradient descent

| Advantages                         | Disadvantages |
| ---------------------------------- | ------------- |
| No need to manually pick $\alpha$  | More complex  |
| Often faster than gradient descent |               |

> These complex algorithms have a "clever inner-loop" called line search algorithm that automatically tries out different values for learning rate $\alpha$ and automatically pick a good one. In fact these algorithms do much more than that.
>
> :older_man:建议不要试图实现这些算法，除非你是数值计算方面的专家，甚至你不必完全理解这些算法就能很好的使用它们。



### 6.7 Multi-class classification: One-vs-all (unfinished)

- How to get logistic regression to work for multi-class classification problems
- One-versus-all classification algorithm

#### Multiclass classification

<!--unfinished-->



## 7 Regularization (unfinished)

> 正则化，是解决（改善）过拟合问题的手段之一

### 7.1 The problem of overfitting

- Explain what is overfitting problem

Example: Linear regression (housing prices)

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524215954.png" alt="image-20210524215953904" style="zoom: 33%;" />

> "Underfit""High bias",高偏差：强行用直线拟合曲线分布的数据，就像“持有偏见，固执认为房价变化就是线性的”，导致拟合结果偏差很大

> "Overfit""High variance",高方差

#### Overfitting

If we have too many features, the learned hypothesis may fit the training set very well ($J(θ)=\frac {1}{2m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})^2\approx0$), but fail to generalize to new examples(predict prices on new examples).

Example: Logistic regression

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210524221122.png" alt="image-20210524221122755" style="zoom:33%;" />

#### Addressing overfitting

If we have la lot of features, and very little training data, then overfitting can become a problem.

Two main options:

1. Reduce number of features
   - Manually select which features to keep (but some feature must be abandoned)
   - Model selection algorithm (later will explain)
2. Regularization
   - Keep all the features, but reduce magnitude/values of parameters $\theta_j$
   - Works well when we have a lot of features, each of which contributes a bit to prediction $y$.

<!--unfinished-->



## 8 Neural Networks: Representation

### 8.1 Non-linear hypothesis

Why we need Neural Networks?

> 当特征很多，线性回归和逻辑回归就不那么好用了。即使他们得出了能够拟合当前样本的结论，该结果也很有可能是过拟合的。

The neural networks which turns out to be a much better way to learn complex nonlinear hypothesis, even when your input feature space (n) is large.

### 8.2 Neurons and the brain

**History of  neural networks**

- Origins: Algorithms that try to mimic the brain.
- WAs very widely used in 80s and early 90s; popularity diminished in late 90s.
- Recent resurgence: State-of-the-art technique for many applications.

**The "one learning algorithm" hypothesis**

- Neuro-rewiring experiments
- Sensor representations in the brain



### 8.3 Model representation I

- How we represent  Neural Networks (hypothesis or model).

#### Neuron model: Logistic unit

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210525151235.png" alt="image-20210525151235213" style="zoom: 67%;" />

**Sigmoid (logistic) activation function**

> 这里的激活函数(activation function)是指代非线性函数$g(z)=\frac1{1+e^{-z}}$的另一个术语

Sometimes we add an extra $x_0$ node (if necessary) called bias unit (偏置单元) or the bias neuron (偏置神经元). It's always equal to 1 so sometime we don't draw it.

In the neural networks literature, the parameters of model $\theta$ is also called **weights of a model**.

#### Neural Network

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210525151843.png" alt="image-20210525151843255" style="zoom:67%;" />

Layer 1: Input layer

Layer 2 : Hidden layer

Layer 3: Output layer

$a_i^{(j)}=$ "activation" of unit $i$ in layer $j$.

$\Theta^{(j)}=$ matrix of weights controlling function mapping form layer $j$ to layer $j+1$

> 可以形象的看作神经网络被这些矩阵（$\Theta^{(j)}$）参数化，因而这些矩阵也被称作权重矩阵（matrix of weights）。权重矩阵控制着从某一层到下一层的映射。

$$
a_1^{(2)}=g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3) \\
a_2^{(2)}=g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3) \\
a_3^{(2)}=g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3) \\
h_\Theta(x)=a_1^{(3)}=g(\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)})
$$

If networks has $s_j$ units in layer $j$, $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1}\times(s_j+1)$.

> The superscript $j$ in parentheses means that these values associated with layer $j$



### 8.4 Model representation II

- How to carry out computation efficiently and show a vectorized implementation.
- Intuition about why these neural network representation

#### Forward propagation: Vectorized implementation

**Define:**

$z_1^{(2)}=\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3$

Thus,

$a_1^{(2)}=z_1^{(2)}$ Similarly, $a_2^{(2)}=z_2^{(2)}$, $a_3^{(2)}=z_3^{(2)}$

We observe that these equations are very much like matrix multiplication. Therefore we try to vectorize the neural network computation.

**Define:**
$$
x=\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ x_3\end{bmatrix} \qquad z^{(2)}=\begin{bmatrix}z_1^{(2)} \\ z_2^{(2)} \\ z_3^{(2)}\end{bmatrix}
$$
Further towards vectorization: 
$$
z^{(2)}=\Theta^{(1)}x=\Theta^{(1)}a^{(1)} \\
a^{(2)}=g(z^{(2)})
$$

>$z^{(2)}$ and $a^{(2)}$ are both 3-dimensional vectors. Function $g$ will process each element in $z^{(2)}$ one by one.

Next, add bias unit: $a_0^{(2)}=1$. Notice that $a^{(2)}\in \mathbb R^4$

$z^{(3)}=\Theta^{(2)}a^{(2)}$

$h_\Theta(x)=a^{(3)}=g(z^{(3)})$

>$z^{(3)}=\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)}$ if you review to the neural networks function.

This process of computing $h(x)$ is also called **forward propagation**, because we start off with the activations of the input-units and then we sort of forward-propagation that to the hidden layer and repeat this process until arriving output layer. The formula we have got is relatively an efficient  way of computing $h(x)$.

#### Neural Network learning its own features

> 如果只关注第二层和第三层，那么神经网络的行为类似于逻辑回归。然而神经网络输出层的输入是隐藏层计算或是说学习的结果（$a_1^{(2)},a_2^{(2)},a_3^{(2)}...$），而不是逻辑回归中初始的特征项$x_1,x_2,x_3...$ 。前者较后者更适合作为假设参数，因此神经网络算法具备灵活快速尝试学习任意特征项，处理更多复杂特征的能力。

#### Other network architectures

The way that neural networks are connected are called the architecture (神经网络的架构). So the architecture refers to how different neurons are connected to each other.

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210525165812.png" alt="image-20210525165812274" style="zoom:50%;" />



### 8.5 Examples and intoitions (unfinished)

- A detail example which shows how a neural network can compute a complex nonlinear function of the input

- Why neural network can be used to learn complex nonlinear hypothesis



## 9 Neural Networks: Learning (unfinished)



## 10 Advice for applying machine learning (unfinished)

10.1 Decide what to try next

10.2 Evaluating a hypothesis

10.3 Model selection and training/validation/test sets



## 11 Machine Learning system design (unfinished)

11.1 Prioritizing what to work on: Spam classification example



## 12 Support Vector Machines (unfinished)

12.1 Optimizaion object

- Sometimes gives a cleaner and a more powerful way of learning complex nonlinear functions



## 13 Clustering

### 13.0 Notation

Train set: $\{x^{(1)},x^{(2)},x^{(3)},...x^{(m)}\}$  (without labels)



### 13.1 K-means algorithm

> K均值算法

K-means is a iterative algorithm. The preparation of the algorithm is to randomly initialize two (depends on how many cluster you want to assign) point, which called the cluster centroids (聚类中心). Then go through each point, detect and record which centre point they are closer to. Second is a move centroid step to the center of all the points in the same group. Repeat these two steps until the grouping of the points no longer changes.

**Input**:

- $K$ (number of clusters)
- Training set $\{x^{(1)},x^{(2)},x^{(3)},...x^{(m)}\}$

$x^{(i)}\in \mathbb R^n$ (drop $x_0=1$ convention)

#### K-means algorithm

Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,...,\mu_K \in \mathbb R^n$

Repeat {
	// <u>cluster assignment step</u>

​	for $i$ = 1 to $m$

​		$c^{(i)}$ := index (form 1 to $K$) of cluster controid close to $x^{(i)}$        (calculate $c^{(i)}$ though $\min_k||x^{(i)}-\mu_k||$)

​	// <u>move centroids step</u>

​	for $k$ = 1 to $K$

​		$\mu_k$ := average (mean) of points assigned to cluster $k$ 
}

> $K$ means the number of centroids and the $k$ means the index of each centriod.
>
> If you have a cluster with no points assigned to it, the usual practice is to delete it.



### 13.2 Optimization objective

- How we can use it to help K-means algorithm find better clusters and avoid local optima.

#### K-means optimization objective

$c^{(i)}$ = index of cluster (1,2,…,$K$) to which example $x^{(i)}$ is currently assigned

$\mu_k$ = cluster centroid $k$ ($\mu_k\in \mathbb R^n$)

$\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

#### Optimization objective:

$$
J(c^{(1)},... c^{(m)},\mu_1,...,\mu_K)=\frac1m\sum^m_{i=1}||x^{(i)}-\mu_{c^{(i)}}||^2 \\
\min_{c^{(1)},... c^{(m)},\mu_1,...,\mu_K} J(c^{(1)},... c^{(m)},\mu_1,...,\mu_K)
$$

> The cost function $J$ is also called discotion function

**Review K-means algorithm**

Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,...,\mu_K \in \mathbb R^n$

Repeat {

​	//minimize $J$ though $c^{(i)}$, $\mu_k$ fixed

​	for $i$ = 1 to $m$

​		$c^{(i)}$ := index (form 1 to $K$) of cluster controid close to $x^{(i)}$

​	//minimize $J$ though $\mu_k$, $c^{(i)}$ fixed

​	for $k$ = 1 to $K$

​		$\mu_k$ := average (mean) of points assigned to cluster $k$ 
}



### 13.3 Randomly initialization

- How to initialize K-means
- How to avoid local optima

> Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,...,\mu_K \in \mathbb R^n$

**Randomly initialization**

Usually, we have $K<m$

Randomsly pick $K$ training examples.

Set $\mu_1,…,\mu_K$ equal to these $K$ examples.

<img src="../../../../../Library/Application Support/typora-user-images/image-20210528155332940.png" alt="image-20210528155332940" style="zoom: 33%;" />

> Local optima we should avoid, but randomly initialization may cause this situation.

> 简而言之，为了避免这种情况的发生，我们可以先初始化一千遍，然后选择畸变最小的那一种情况，也就是最有潜力，最不可能陷入局部最优的情况来进行接下来的运算。



### 13.4 Choosing the number of cluster

- Manual
- Elbow method
- Later downstream purpose



## 14 Dimensionaliy Reduction (unfinished)

14.1 Motivation I: Data Compression

14.2 Motivation II: Visualization

14.3 Principle Component Analysis problem formulation (PCA)

> 主成分分析法

- Compression algorithm

14.4 Principle Component Analysis algorithm

Data preprocessing



## 15 Anomaly Detection (unfinished)

15.1 Problem motivation



## 16 Recommeder Systems (unfinished)



## 17 Large Scale Machine Learning (unfinished)

### 17.1 Learning with large datasets

$$
\theta_j :=\theta_j -\alpha\frac {1}{m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})x_j^{(i)}
$$

> ​	对于上亿的数据规模来说，计算梯度下降中的求和函数是难以承受的负担



### 17.2 Stochastic gradient descent

#### Review linear regression with gradient descent

<!-- unfinished -->

#### Batch gradient descent

#### Stochastic gradient descent



### 17.3 Mini-batch gradient descent







## Markdown

\begin{bmatrix}\end{bmatrix}

\mathbb R

[公式](https://www.jianshu.com/p/25f0139637b7)

[公式2](https://www.jianshu.com/p/e74eb43960a1)

[语法](https://www.jianshu.com/p/191d1e21f7ed)

























