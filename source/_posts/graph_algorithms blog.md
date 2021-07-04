---
title: Graph Algorithms
date: 2021-07-04
tag: Data structure and algorithm
---



# Graph Algorithms


## 1 最小生成树问题
### 1.1 Prim Algorithm

寻找加权连通图里搜索最小生成树。意即由此算法搜索到的边子集所构成的树中，不但包括了连通图里的所有顶点，且其所有边的权值之和亦为最小。

#### [基本思想](https://blog.csdn.net/luoshixian099/article/details/51908175)

此算法可以称为“加点法”，每次迭代选择代价最小的边对应的点，加入到最小生成树中。算法从某一个顶点s开始，逐渐长大覆盖整个连通网的所有顶点。

1. 图的所有顶点集合为VV；初始令集合u={s},v=V−uu={s},v=V−u;

2. 在两个集合u,vu,v能够组成的边中，选择一条代价最小的边(u0,v0)(u0,v0)，加入到最小生成树中，并把v0v0并入到集合u中。

3. 重复上述步骤，直到最小生成树有n-1条边或者n个顶点为止。

   > 由于不断向集合u中加点，所以最小代价边必须同步更新；需要建立一个辅助数组closedge，用来维护集合v中每个顶点与集合u中最小代价边信息

#### Time efficiency

>  记顶点数$v$，边数$e$

邻接矩阵: $O(v^2) $

邻接表: $O(e log_2v)$



### 1.2 Kruskal Algorithm

> 克鲁斯克尔算法 

#### 简述

寻找最小生成树的算法，Kruskal算法在图中存在相同权值的边时也有效。

#### 基本思想

此算法可以称为“加边法”，初始最小生成树边数为0，每迭代一次就选择一条满足条件的最小代价边，加入到最小生成树的边集合里。
1. 把图中的所有边按代价从小到大排序；
2. 把图中的n个顶点看成独立的n棵树组成的森林；
3. 按权值从小到大选择边，所选的边连接的两个顶点$u_i$,$v_iu_i$,$v_i$,应属于两颗不同的树，则成为最小生成树的一条边，并将这两颗树合并作为一颗树。
4. 重复(3),直到所有顶点都在一颗树内或者有n-1条边为止。

#### Time efficiency

> $e$为图中的边数

$elog_2e$ 

<!-- more -->



## 2 最短路径问题

### 2.1 Dijkstra Algorithm

> 迪杰斯特拉算法

#### 简述

迪杰斯特拉(Dijkstra)算法是典型最短路径算法，用于计算一个节点到其他节点的最短路径。它的主要特点是以起始点为中心向外层层扩展(广度优先搜索思想)，直到扩展到终点为止。

![Dijkstra_Animation](https://raw.githubusercontent.com/zhuyihapi/picture/main/20210610104853.gif)

#### 基本思想

1. 通过Dijkstra计算图G中的最短路径时，需要指定起点s(即从顶点s开始计算)。
2. 此外，引进两个集合S和U。S的作用是记录已求出最短路径的顶点(以及相应的最短路径长度)，而U则是记录还未求出最短路径的顶点(以及该顶点到起点s的距离)。
3. 初始时，S中只有起点s；U中是除s之外的顶点，并且U中顶点的路径是”起点s到该顶点的路径”。然后，从U中找出路径最短的顶点，并将其加入到S中；接着，更新U中的顶点和顶点对应的路径。 然后，再从U中找出路径最短的顶点，并将其加入到S中；接着，更新U中的顶点和顶点对应的路径 … 重复该操作，直到遍历完所有顶点。



### 2.2 Bellman–Ford Algorithm

> [贝尔曼-福特算法](https://zh.wikipedia.org/wiki/%E8%B4%9D%E5%B0%94%E6%9B%BC-%E7%A6%8F%E7%89%B9%E7%AE%97%E6%B3%95)

### 简述

其优于迪科斯彻算法的方面是边的权值可以为负数、实现简单，缺点是时间复杂度过高，高达$O(|V||E|)$。但算法可以进行若干种优化，提高了效率。

贝尔曼-福特算法与迪科斯彻算法类似，都以松弛操作为基础，即估计的最短路径值渐渐地被更加准确的值替代，直至得到最优解。在两个算法中，计算时每个边之间的估计距离值都比真实值大，并且被新找到路径的最小长度替代。 然而，迪科斯彻算法以贪心法选取未被处理的具有最小权值的节点，然后对其的出边进行松弛操作；而贝尔曼-福特算法简单地对所有边进行松弛操作，共$|V|-1$次，其中$|V|$是图的点的数量。在重复地计算中，已计算得到正确的距离的边的数量不断增加，直到所有边都计算得到了正确的路径。这样的策略使得贝尔曼-福特算法比迪科斯彻算法适用于更多种类的输入。



### 2.3 Floyd-Warshall Algorithm

> 弗洛伊德算法

#### 简述

弗洛伊德算法是解决任意两点间的最短路径的一种算法，可以正确处理有向图或负权（但不可存在负权回路）的最短路径问题，同时也被用于计算有向图的传递闭包。



#### Time efficiency

时间复杂度：$O(N^3)$

空间复杂度：$O(N^3)$



### 2.4 SPFA Algorithm

> Moore-Bellman-Ford 算法

#### 简述

它的原理是对图进行V-1次松弛操作，得到所有可能的最短路径。其优于迪科斯彻算法的方面是边的权值可以为负数、实现简单，缺点是时间复杂度过高，高达 O(VE)。但算法可以进行若干种优化，提高了效率。

#### 基本思想

我们用数组dis记录每个结点的最短路径估计值，用邻接表或邻接矩阵来存储图G。我们采取的方法是动态逼近法：设立一个先进先出的队列用来保存待优化的结点，优化时每次取出队首结点u，并且用u点当前的最短路径估计值对离开u点所指向的结点v进行松弛操作，如果v点的最短路径估计值有所调整，且v点不在当前的队列中，就将v点放入队尾。这样不断从队列中取出结点来进行松弛操作，直至队列空为止

我们要知道带有负环的图是没有最短路径的，所以我们在执行算法的时候，要判断图是否带有负环，方法有两种：

1. 开始算法前，调用拓扑排序进行判断（一般不采用，浪费时间）

2. 如果某个点进入队列的次数超过N次则存在负环（N为图的顶点数）

   

## Other

### Dynamic Programming

#### Definition

Dynamic Programming (DP) is an algorithmic technique for solving an optimization problem by breaking it down into simpler subproblems and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems.

#### Steps for Solving DP Problems

1. Define subproblems

2. Write down the recurrence that relates subproblems
3. Recognize and solve the base cases



### Greedy Method

#### Definition

A greedy algorithm is an algorithmic strategy that makes the best optimal choice at each small stage with the goal of this eventually leading to a globally optimum solution. This means that the algorithm picks the best solution at the moment without regard for consequences. It picks the best immediate output, but does not consider the big picture, hence it is considered greedy.

A greedy algorithm works by choosing the best possible answer in each step and then moving on to the next step until it reaches the end, without regard for the overall solution. It only hopes that the path it takes is the globally optimum one, but as proven time and again, this method does not often come up with a globally optimum solution. In fact, it is entirely possible that the most optimal short-term solutions lead to the worst possible global outcome.



### Divide and Conquer

### Definition

The divide-and-conquer paradigm is often used to find an optimal solution of a problem. Its basic idea is to decompose a given problem into two or more similar, but simpler, subproblems, to solve them in turn, and to compose their solutions to solve the given problem. Problems of sufficient simplicity are solved directly.



### Backtracking

#### Definition

Backtracking is a general algorithm for finding all (or some) solutions to some computational problems, notably constraint satisfaction problems, that incrementally builds candidates to the solutions, and abandons a candidate ("backtracks") as soon as it determines that the candidate cannot possibly be completed to a valid solution.
