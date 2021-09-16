# Machine Learning 笔记

结合ECE-6143 ML课程的笔记与周志华的西瓜书。课上的笔记已经upload到github了。这里就按照周志华的西瓜书做笔记学习，笔记之中也会加上一些课上的笔记，这样更容易整合，还结合Greedy的课程，就会非常的全面。

## Unit 1 Data processing

### 1.1	Feature scaling

https://zhuanlan.zhihu.com/p/34480619

## Unit 2	模型评估与选择

### 2.1	Empirical Error and Overfitting

我们先了解一些概念：

* error rate：就是把分类错误的样本数占样本总数的比例。比如说，总共有m个样本，然后有a个分类错误，那么error rate = a/m。
* accuracy：就是分类正确的样本数占样本总数的比例。也是1 - error rate。
* error：模型的预测值与真实值之间的差异，在训练集上的error，就是training error，或者是empirical error。在新样本上的error，就是generalization error。

我们想要的是什么？就是generalization error要尽可能小，但是由于不知道新样本是什么情况，就只能看training error。这里呢，又有两个概念需要理解，就是overfitting and underfitting。下面这张图就很好的诠释了这两个概念。

* overfitting：就是学的太过了，把不太一般的特性都给学到了
* underfitting：学的不够

<img src="https://user-images.githubusercontent.com/68700549/120700323-8408ec00-c47f-11eb-9f47-a7266eea1c0a.png" alt="1__7OPgojau8hkiPUiHoGK_w" style="zoom: 67%;" />

underfitting容易解决一点，而overfitting比较麻烦，做ML，DL就一定会面临这个问题。

### 2.2	评估方法

我们很难得到generalization error，就只能把model用在testing set得到的error作为generalization error的近似值。这里我们说一个validation set。就是我们分数据集的时候，我们可能会分三个，training, validation, and testing set. validation set的主要作用是帮助调参，就是说我们在training set中进行训练，然后看看在validation set的performance，然后调参。直到我们有了一个好的model之后，再去看看在testing set的performance。testing set一般是在最后的最后才使用。我们有一个大的数据集，下面就是一些常用的做法来分training set and testing set

#### 2.2.1	Hold out

就是很简单的三七分，二八分。把数据集直接拆分为training set and testing set。我们要保证这两个数据集同分布就行，如果样本差异很大就不行。这个方法呢，有一个窘境，就是如果training set较大，testing set较小，那么训练时就很接近用整个数据集进行训练了，这样在testing set出来的结果可能就不太稳定。但是如果让testing set大一点，那么因为training set相对较小，可能跟整体的数据集有一定的差异，这样训练结果可能会降低保真性(fidelity). 如果从方差-偏差的角度来想，就是当testing set较小的的时候，方差（variance）比较大（因为不稳定）。testing set较大的时候，偏差（bias）比较大。目前没有一个比较好的解决方法，一般就是三七分，二八分。

####	2.2.2	Cross Validation

就是所谓的cross validation。这个非常简单，也经常使用，在数据集上，训练的时候进行划分，比如说k-fold，那么训练就用k-1个training set，然后1个testing set。k一般取5或10. 缺点就是数据集大的时候，计算量大。cross validation中，有一个叫Leave-one-out，就是当k=m时，m是整个数据集的大小。Leave-one-out被认为是评估比较准确的一个方法，但是计算量及其大。

<img src="https://user-images.githubusercontent.com/68700549/120706837-95ee8d00-c487-11eb-8677-f822e4d9ef39.png" alt="1_AAwIlHM8TpAVe4l2FihNUQ" style="zoom: 33%;" />

Cross validation还有可以用来调参，其实就是brute force。意思就是假设我有一个参数k，取值是[0,1,2,3,4,5].我们要看这个k取什么值最好，那就每个值取一遍，然后都run k-fold cross validation。最后看哪个结果比较好，就取哪个值。

####	2.2.3	Bootstrapping

就是之前的方法中，总有一部分留出来作为testing set，然后training set就跟整体的数据集有一个差异。Leave-one-out很好但是计算量大，怎么办呢，就有了Bootstrapping。Bootstrap sampling可以有效地减少训练样本规模不同造成的影响。方法很简单，就是有放回的sampling。假设整个数据集有n个data，取一个，然后放回去，然后再从这n个data中再取一个，重复n次，这样我们就有了一个新的数据集，同样包含m个数据。我们可以做一个计算，就是计算那些没有被sampling到的样本大概还会留下多少。样本在n次采样中始终不被采到的概率是$1-\frac{1}{n}$, 重复n次，得到$(1-\frac{1}{n})^n$,取极限，我们就可以得到一个值，大概是0.368. 也就是说，大概会有36.8%的数据会没被采到。我们就可以把这36.8%的数据作为testing set，然后刚刚得到的新数据（有n个data）作为training set。下面这张图就是解释

<img src="https://user-images.githubusercontent.com/68700549/123884934-631aa600-d91a-11eb-992d-6e2ca70dd790.png" alt="WeChat Screenshot_20210629204124" style="zoom:50%;" />

Bootstrapping呢，在数据集较小，难以有效划分training、testing set时会比较有用。但是有一个缺点，因为是有放回，采样时会有重复，新的数据集会跟原来的分布不大一样，会有一个估计偏差。因此呢，在数据量足够的时候，就用以上两种方法（hold-out, cross validation）

###	2.3	Performance Measure

就是performance measure。当我们有了一个model之后，对这个model的generalization能力进行评估。

regression当中最常使用的就是mean squared error(MSE). 计算公式也非常简单。$E=\frac{1}{m}\sum_{i=1}^m (f(x_i)-y_i)^2$.

下面就主要介绍分类任务中常用的performance measure

####	2.3.1	Error rate and accuracy

这两个是classification中最常用的performance measurement。在binary classification and muti-classification 中都适用。之前有提到，公式也非常简单，就看与真实值是否一样即可。

####	2.3.2	Precision, Recall and F1

对于一些需求，光有Error rate and accuracy是不够的。因为这只能是用来看判别率。但是有一些问题，比如说，在Web搜索方面的应用，“检索出的信息有多少比例是用户感兴趣的”，“用户感兴趣的信息中有多少是被检索出来的”，这个时候就需要引入precision and recall了。

对于分类问题，我们是可以得到confusion matrix的。比如说，下面这张图，就是confusion matrix。

<img src="https://user-images.githubusercontent.com/68700549/120716598-42367080-c494-11eb-89ee-e95bff2d4eae.png" alt="cm_colored_1-min" style="zoom:50%;" />

我们首先要去理解TN, FP, FN, TP. 

需要重点理解的是，在同一个系统中，若TP增加，则FP也增加，下图表示了这个关系

<img src="https://user-images.githubusercontent.com/68700549/123738721-55f6ac00-d873-11eb-90cb-2b2b485c14f6.png" alt="WeChat Screenshot_20210629004528" style="zoom:50%;" />

* Precision：在所有被预测为“positive”的label中，真正是positive的比例 $P=\frac{TP}{TP+FP}$
* Recall：所有positive的数据中，被预测为positive的比例 $R=\frac{TP}{TP+FN}$

这两个是一对矛盾的度量，一般来说，precision高时，recall比较低。precision低时，recall比较高。为啥呢？因为是threshold的影响。随着threshold的增加，precision会增高，recall会下降

<img src="https://user-images.githubusercontent.com/68700549/120719842-fc2fdb80-c498-11eb-9d6b-fe18d9e4c18f.png" alt="WeChat Screenshot_20210603182412" style="zoom: 67%;" />

我们一般会作一个P-R曲线。就是根据学习器的预测结果对样例进行预测，得到结果（得到的结果是一个概率值，设置threshold之后，才得到是哪一个分类。）根据这个概率，从大到小进行排序。然后以recall作为x轴，precision作为y轴，得到P-R曲线。代码如下。

```python
#coding:utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
#y_true为样本实际的类别，y_scores为样本为正例的概率
y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
y_scores = np.array([0.9, 0.75, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
#print(precision)
#print(recall)
#print(thresholds)
plt.plot(recall,precision)
plt.show()
```

当我们有多个model的时候，就要进行选择，比如说，我们要下面这张图，怎么看哪个model比较好呢？如果A曲线能把B曲线给包住，那么A model就是比较好的，比如说橙色线就能把蓝色线给包住，那么橙色线代表的那个model就比较好。那如果有交叉呢？那么就看Break-Even-Point。就是当precision=recall时，比较大的值是比较好的。比如说黑色的线与橙色的线，黑色的Break-Even-Point就比较大，说明比较好，因为希望两者都越大越好。

<img src="https://user-images.githubusercontent.com/68700549/120717549-bd4c5680-c495-11eb-9442-d6acf292f5ce.png" alt="WeChat Screenshot_20210603180128" style="zoom: 67%;" />

光看Break-Even-Point也不够，还要看看F1-Score，公式中的2是加权平均. 我们更倾向于选F1-Score较小的model，但是F1-Score较大也代表着model的稳定
$$
F1 = \frac{2\times P \times R}{P+R}=\frac{2\times TP}{样例总数 + TP - TN}
$$
有时我们也得看需求选precision and recall。比如说，推荐用户感兴趣的内容，我们希望准一点，这是需要precision高。又比如，抓逃犯，我们希望不漏过一个，就需要recall高。



####	2.3.3	ROC and AUC

* ROC: Receiver Operating Characteristic. 主要作用就是衡量分类器的分类能力
* AUC: Area Under ROC Curve

绘图方法也是一样，对预测值要进行排序，不断地调threshold，confusion matrix就会发生改变，那么值也会相对应地改变，于是就可以绘图了。ROC的纵轴是True Positive Rate (TPR). 横轴是 False Positive Rate (FPR)

* $TPR=\frac{TP}{TP+FN}$. 在所有positive的label中，看看预测是positive的概率
* $FPR=\frac{FP}{TN+FP}$。 在所有negative的label中，看看预测是positive的概率

我们希望TPR越大越好，FPR越小越好。因为TPR是预测正确的，而FPR是预测错误的。

```Python
import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = # ground truth labels
y_probas = # predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
```

ROC curve是越靠左上角就越好，AUC也越大越好。因为TPR就越大，FPR就越小。

<img src="https://user-images.githubusercontent.com/68700549/120736603-88062f80-c4ba-11eb-93e4-d3dc5f0cd988.png" alt="WeChat Screenshot_20210603222440" style="zoom: 50%;" />

错误率是在阈值固定的情况下得出的，ROC曲线是在阈值随着样本预测值变化的情况下得出的。ROC曲线上的每一个点，都对应着一个错误率。

####	2.3.4	代价敏感错误率与代价曲线

可以去了解一下cost curve

https://www.zhihu.com/question/63492375

###	2.4	比较检验

就是ML得出的model，我们希望比较的是generalibility, 我们之前是一直在testing set做估计，然后这并不代表model就有很好的泛化能力。因为model在testing set的performance跟testing set的选择是有很大的关系的，且每次的选择啊，大小啊，都会有影响，结果也会不同。再者，ML model有一定的随机性，就是说，即使参数相同，在同一个testing set上跑，每次的结果也可能不同。下面就介绍可以检验ML model的泛化性能的方法。做一些了解即可，知道有这个方法就行。主要了解假设检验

####	2.4.1	假设检验

####	2.4.2	交叉验证t检验

####	2.4.3	McNemar检验

####	2.4.4	Friedman检验 and Nemenyi后续检验

###	2.5	Variance and Bias

关于这个，我们就需要了解，泛化误差=偏差+方差+噪声，公式就是
$$
E=bias^2(x)+var(x)+\epsilon ^2
$$
然后，我们需要了解的是variance and bias是有冲突的，比如说，训练不足，拟合能力不够强，也就是所谓的打不中目标，这是bias主导了泛化误差。如果学习过度，拟合能力太强，不该拟合的也拟合上去了，这就导致不稳定，这是variance主导了泛化误差。

Bias就是期望值与真实值的差异，随着模型的复杂度增加，bias减小，因为模型复杂之后，更容易缩小预测值与真实值的差异。

Variance就是在一个特定的数据集上训练的模型与所有模型的期望的差异。衡量模型对特定数据集的变化的敏感度。随着模型的复杂度增加，Variance增加。数据集一变，可能结果就很不一样。

下面这张图就很好地解释了variance and bias。

<img src="https://user-images.githubusercontent.com/68700549/121074278-a3668880-c7a1-11eb-8cff-cca9747b5a4b.png" alt="WeChat Screenshot_20210607150352" style="zoom:67%;" />

关于泛化误差

<img src="https://user-images.githubusercontent.com/68700549/121074420-d3ae2700-c7a1-11eb-916a-71fa9180fac2.png" alt="WeChat Screenshot_20210607150529" style="zoom:67%;" />



##	Unit 3	线性模型

线性模型比较简单，重点去理解linear regression and logistics regression，这个比较简单，可以回去看ECE-GY 6143 ML笔记

### 3.1	Linear Regression

Linear Regression比较简单，就是通过属性的线性组合来进行预测的函数，方程是
$$
f(x)=w_1x_1+w_2x_2+...+w_dx_d+b
$$
一般就写成向量的模式，也就是
$$
f(x)=w^Tx+b\\
w=(w_1;w_2;...;w_d)
$$
那怎么求$w,b$呢？Linear Regression试图得到$f(x_i)=wx_i+b\simeq y_i$. 所以Linear Regression的主要任务就是减小$f(x_i)$与$y_i$​之间的差距，这里呢，Mean Square Error(MSE) 就是这里常用的性能度量，所以，需要最小化MSE。

这里谈一谈回归的几种loss function

* MSE

  Mean square error，也叫L2 loss，是我们最常用的loss function之一。它的公式是
  $$
  MSE=\frac{\sum_{i=1}^n (y_i-\hat{y}_i)^2}{n}
  $$
  我们看下图的例子，假设true value是100，横坐标是predictive value，纵坐标是MSE

  <img src="https://user-images.githubusercontent.com/68700549/126798926-811c97a1-2790-48fb-bc28-6b8af796d078.png" alt="v2-1f59a4aa5abb6ee0d9eeb0d60ffe2a23_r" style="zoom: 67%;" />

* MAE

  Mean absolute error，也叫L1 loss。是true value跟predictive value之间的绝对值之和。公式就是
  $$
  MAE=\frac{\sum_{i=1}^n |y_i-\hat{y}_i|}{n}
  $$
  <img src="https://user-images.githubusercontent.com/68700549/126799372-8ec6d498-e62e-469a-83bb-3ba7d221f504.png" alt="v2-203eb790c821f05d1f7209acae68d2e4_r" style="zoom:67%;" />

  这里，我们对比一下MAE和MSE

  使用MSE更容易解决问题，但是使用MAE对于outlier更加robust。因为MSE是平方嘛，如果数据中outlier较多，我们在使用model的时候，不怎么在意outlier，但是outlier又还在数据中，我们就用MAE。如果我们不想要outlier，首先应该先尝试remove outlier，然后用MSE。因为有outlier的存在，MSE的performance会比较差

  https://zhuanlan.zhihu.com/p/39239829

  MAE有一个缺点，就是它是直线嘛，不管对于什么误差，都是一样的梯度，即使对于非常小的误差，也是一样的梯度，不好收敛，这时就要考虑加入动态学习率，而且，在等于0的地方不可导。MSE就不存在这个问题，当损失值小时，梯度也会很小。如下图所示

  <img src="https://user-images.githubusercontent.com/68700549/126830391-21ff0391-127c-43cd-9bf4-844c4bd79209.jpg" alt="v2-c22f272fd23056312d13f4dc47718ce6_r" style="zoom:67%;" />

  这两个loss function存在的问题

  可能会出现两种损失函数都无法给出理想预测值的情况。例如，如果我们的数据中90% 的观测值的目标真值为150， 剩余10%的目标值在0-30之间。那么存在MAE损失的模型可能会预测全部观测值的目标值为150，而忽略了那10%的异常情况，因为它会试图趋向于中间值。在同一种情况下，使用MSE损失的模型会给出大量值范围在0到30之间的预测值，因为它会偏向于异常值。

  这时我们就要考虑Huber loss了

  ```python
  # true: 真目标变量的数组
  # pred: 预测值的数组
  
  def mse(true, pred): 
      return np.sum((true - pred)**2)
   
   def mae(true, pred):
    return np.sum(np.abs(true - pred))
   
   # 在 sklearn 中同样适用
   
   from sklearn.metrics import mean_squared_error
   from sklearn.metrics import mean_absolute_error
  ```

  

* Huber loss

  相比MSE，Huber loss会对outlier是没那么敏感。HUber

  loss在值为0的时候也是可导的。Huber loss的公式如下
  $$
  L_\delta (y,f(x))=\left\{
  \begin{array}{rcl}
  \frac{1}{2}(y-f(x))^2       &      & {|y-f(x)|\le \delta}\\
  \delta |y-f(x)|-\frac{1}{2}\delta^2     &      & \text{otherwise}
  \end{array} \right.
  $$
  图像如下图所示，不同颜色的线代表不同的$\delta$

  <img src="https://user-images.githubusercontent.com/68700549/126831115-30bc24e9-aa80-4ce1-b807-5144f4f73e61.png" alt="v2-9015dea99bcca5f54cb18af2d0286ce6_r" style="zoom:67%;" />

  Huber相当于结合了MSE和MAE，不会说梯度始终很大，对outlier也比较robust。但是，这里有个参数$\delta$​，它的值非常关键，需要训练。

  ```python
  # huber 损失
  def huber(true, pred, delta):
      loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
      return np.sum(loss)
  ```

  

* Log cosh loss

  Log-cosh比L2 loss 更平滑，公式是
  $$
  L(y,\hat{y})=\sum_{i=1}^n log(cosh(\hat{y}_i -y_i))
  $$
  下面这个是图像

  <img src="https://user-images.githubusercontent.com/68700549/126832035-5fa46f7f-e7d5-4986-974d-f8da0686c5bc.png" alt="v2-5507fc4e82b44ddf974237ee5ab0fe7b_r" style="zoom:67%;" />

  它的优点是：对于较小的X值，log(cosh(x))约等于(x ** 2) / 2；对于较大的X值，则约等于abs(x) - log(2)。这意味着Log-cosh很大程度上工作原理和平均方误差很像，但偶尔出现错的离谱的预测时对它影响又不是很大。它具备了Huber损失函数的所有优点，但不像Huber损失，它在所有地方都二次可微。二次可微是可用于XGBoost，但是如果用于XGBoost，如果始终出现非常大的偏离目标的预测值时，它就会遭受梯度问题，因此会导致XGboost的节点不能充分分裂。

  ```python
  # log cosh 损失
  def logcosh(true, pred):
      loss = np.log(np.cosh(pred - true))
  	return np.sum(loss)
  ```

  

* Quantile loss



#### 3.1.1	Generalized linear model

GLM

### 3.2	Logistic Regression

这个处理的是分类问题，Logistic regression输出的是概率。公式就是$f(z)=\frac{1}{1+e^{-z}},z=w^Tx+b$. 图像如下图所示

<img src="https://user-images.githubusercontent.com/68700549/126342571-f2516e69-2b24-47a8-8513-d0d80294b4e0.png" alt="WeChat Screenshot_20210720103259" style="zoom: 33%;" />

我们这里要了解几个公式

| 类别                   | 公式                                                         |
| ---------------------- | ------------------------------------------------------------ |
| 类别为1的概率          | $P(1|x)=\frac{1}{1+e^{-(w^Tx+b)}}$​                           |
| 类别为0的概率          | $P(0|x)=1-P(1|x)=1-\frac{1}{1+e^{-(w^Tx+b)}}=\frac{1}{1+e^{w^Tx+b}}$ |
| 类别1与0概率比值       | $\frac{P}{1-P}=e^{w^Tx+b}$                                   |
| 类别1与0比值的自然对数 | $ln\frac{P}{1-P}=w^Tx+b$                                     |

接下来我们看loss function，这是convex function，可以得到global minima
$$
\begin{matrix}
P(Y=1|x;w)=f(x;w)=\frac{1}{1+e^{-(w^Tx+b)}}\\
J(w)=-\sum_{i=1}^N y^{(i)}ln(P(Y=1|X=x^{(x_i)};w))+(1-y^{(i)})ln(1-P(Y=1|X=x^{(i)};w))
\end{matrix}
$$
$y^{(i)}$​要么取0，要么取1. 因为有两个类别，要么是1，要么是0，所以两者都需要考虑。加个负号的原因是要求最小值。

我们要求导从而求$w$, 我们logistic regression的概率公式是$f(x;w)=\frac{1}{1+e^{-(w^Tx+b)}}$​，这是类别为1的概率

loss function也已经知道了，$\alpha$是learning rate，求导就是
$$
\begin{matrix}
\bigtriangledown_{w}J(w)=\sum_i x^{(i)}(f(x^{(i)};w)-y^{(i)})\\
w=w-\alpha\bigtriangledown_{w}J(w)
\end{matrix}
$$
现在，我们来理解一下这个$w$​和odds(就是类别1和类别0的比值)的意义，自己看上面的公式。这里讲个例子

假设有如下数据

| age（$x_1$） | Annual Salary（$x_2$） | if buy car (y) |
| ------------ | ---------------------- | -------------- |
| 20           | 3                      | 0              |
| 23           | 7                      | 1              |
| 31           | 10                     | 1              |
| 42           | 13                     | 1              |
| 50           | 7                      | 0              |
| 60           | 5                      | 0              |

通过训练，得到$w_1=-0.2,w_2=0.92,b=-0.04$.

系数$w_2=0.92$意味着，如果年收入增加一万，一个人买车和不买车的概率比值与之前相比较，增加$e^{0.92}=2.5$倍。

系数$w_1=-0.2$​意味着，如果年年龄增加一岁，一个人买车和不买车的概率比值与之前相比较，降低$e^{-0.2}=0.82$​​倍。

如果logistic regression要进行多分类，那么，我们就要用softmax要进行求概率，公式就是
$$
P(y=c_i|x;w)=\frac{exp(w^{(i)^T}x+b_i)}{\sum_{j=1}^K exp(w^{(j)^T}x+b_j)}
$$
Multiclass的loss function就是
$$
\begin{matrix}
L(w)=-\sum_{k=1}^K \mathbb{1}\{y=k\}log\hat{p}(y=k|x)\\
=-\sum_{k=1}^K \mathbb{1}\{y=k\}log\frac{exp(w^{(k)^T}x+b_k)}{\sum_{j=1}^K exp(w^{(j)^T}x+b_j)}
\end{matrix}
$$
最后求导也是一样的

这里也谈一谈分类的几种loss function

* Log loss
* Focal loss
* KL divergence
* Exponential loss
* Hinge loss

#### 3.2.1	Kernel Logistic Regression

### 3.3	LDA

Linear Discriminant Analysis. LDA有两个作用，一个是classification，另一个是reduce dimension。主要就是reduce dimension用。当然，LDA也是可以参考SVM那样的，就是用kernel投射到高维，再进行降维，或者是classification。

我们先用二维的来解释，LDA非常简单，就是有训练数据，我们要把训练数据投影到一条直线上，如下图所示。投影之后要使得相同的样例的投影点尽可能接近，不同类别的投影簇要尽可能远离

<img src="https://user-images.githubusercontent.com/68700549/123687961-7eed5180-d81f-11eb-878b-f9890703964d.png" alt="WeChat Screenshot_20210628144513" style="zoom: 33%;" />

数学推导就是，我们先考虑二分类，假设有数据集$D=\{(x_i,y_i)\}_{i=1}^m,y_i\in \{0,1\}$，令$X_i,\mu_i,\Sigma_i$分别表示第$i\in \{0,1\}$的数据集合、均值向量、covariance matrix。我们先假设有这么一条直线$w$.因为LDA的目标就是要找一条直线，把数据给投影过去。假设找到直线后，这两类的样本的中心在直线上的投影分别是$w^T\mu_0$和$w^T\mu_1$.

这样两个样本之间的距离就是$D(X_0,X_1)=||w^T(\mu_1-\mu_2)||_2^2$,我们要让这个距离尽可能大。

因为要是同类的所有点要尽可能在一块，所以，对于同类的数据，方差要尽可能小。我们先求这两类投影后的方差，variance
$$
Var(X'_0)=\sum_{x\in X_0}(w^Tx-w^T\mu_0)^2\\
Var(X'_1)=\sum_{x\in X_1}(w^Tx-w^T\mu_1)^2
$$
于是，我们可以得到一个优化目标，也就是
$$
J(w)=\frac{D(X_0,X_1)}{Var(X'_0)+Var(X'_1)}\\
=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{\sum_{x\in X_i}w^T(x-\mu_i)(x-\mu_i)^Tw}
$$
现在，定义两个matrix，一个是within-class scatter matrix，也就是$S_w=\sum_{x\in X_i}(x-\mu_i)(x-\mu_i)^T$, 还有一个matrix是between-class scatter matrix, 也就是$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$

于是，我们的优化目标就可以写成,这也是LDA需要求的最大化目标，就是$S_b$与$S_w$的generalized Rayleigh quotient。
$$
J(w)=\frac{w^TS_bw}{w^TS_ww}
$$
之后的过程有点复杂，到这里就可以了，我们只需要知道后面可以根据这个优化目标找出一个条件等价式，然后用Lagrange multiplies得出$S_bw=\lambda S_ww$即可。这个式子是可以化为$S_bw=\lambda(\mu_0-\mu_1)$.不用知道怎么变得，也不用知道为什么。

然后我们就可以得出$w=S_w^{-1}(\mu_0-\mu_1)$. 这个$w$就是我们要找的那条投影线。$S_w^{-1}$是通过SVD来得到的，$S_w=U\Sigma V^T,S_w^{-1}=V\Sigma^{-1}U^T$.

当然，LDA也是可以用在多分类的，multi-classification的。

总结：LDA的dimension reduction的过程(multi-classification)就是

* 计算每个类别的均值$\mu_i$,全局样本均值$\mu$
* 计算within-class scatter matrix $S_w$, and between-class scatter matrix $S_b$, 还有全局散度矩阵$S_t=S_b+S_w$
* 对$S_w^{-1}S-b$做SVD
* 取最大的$d'$个特征值所对应的特征向量
* 计算投影矩阵

PCA与LDA的比较

* 从目标出发，
  * PCA选择的是投影后数据方差最大的方向。由于它是无监 督的，因此PCA假设方差越大，信息量越多，用主成分来表示原始数据可以去除 冗余的维度，达到降维。
  * LDA选择的是投影后类内方差小、类间方差大的方 向。其用到了类别标签信息，为了找到数据中具有判别性的维度，使得原始数据 在这些方向上投影后，不同类别尽可能区分开。

* 从应用的角度，我们可以掌握一个基本的原则——对无监督的任务使用PCA 进行降维，对有监督的则应用LDA。

### 3.4	Class imbalance

### 3.5	Gradient Descent

gradient呢，就是函数的导数方向。gradient decent是求解无约束多元函数极值的常用方法。在求解目标函数$f(x)$的最小值时，如果目标函数是无约束的凸函数，就可以使用gradient descent。我们的目标也是将cost function最小化

#### 3.5.1	Batch Gradient Descent (BGD)

这个是最原始的方法，就是每一次更新参数的时候，把所有的样本都用上，假设有m个样本，就对每个样本都进行求一个$\theta$的变化，然后取均值，再进行更新$\theta$.

优点就是容易找到global minima，可以并行实现。缺点就是当样本数目多是，训练过程会很慢，当样本量小的时候可以用

如果function是convex function，一定会找到最优解。如果是非凸，一定能找到局部最优解。不适合online的情况

online machine learning is a method of machine learning in which data becomes available in a sequential order and is used to update the best predictor for future data at each step, as opposed to batch learning techniques which generate the best predictor by learning on the entire training data set at once. 

就是说没有一次性把所有data都提供给你，后来陆陆续续地还给你一部分data。BGD不适合online的原因是它是对整体求均值的，加入新数据后，均值就变了，所以不适合online情况。

#### 3.5.2	Stochastic Gradient Descent (SGD)

每更新一次参数就使用一个样本，都是随机选择的样本，更新很多次。如果样本量很大，比如说有几十万，那么可能只用其中几万条就可以将$\theta$的值迭代到最优解了。BGD的计算量太大，SGD就减少了很多的计算量。但是SGD就是对噪音敏感，使得SGD并不是每次iteration都是朝着最优化的方向走。优点就是训练速度快，但是缺点也很明显，准确度下降，并不是全局最优，不容易并行实现。就是SGD在搜索过程中比较盲目。就比如下图，会比较乱

<img src="https://user-images.githubusercontent.com/68700549/123549311-779e4900-d736-11eb-8816-e63e221b9625.png" alt="WeChat Screenshot_20210627105716" style="zoom:50%;" />

SGD适合online情况，能够帮助跳出saddle point。

#### 3.5.3	Mini-batch Gradient Descent (MBGD)

就是把上面的两个方法结合一下，训练速度快，还保证了准确率.就是每次更新$\theta$时，随机取一部分样本(常用值：32,64)来进行更新

## Unit 4	Decision Tree

### 4.1	基本流程

决策树非常的简单，如下图所示

<img src="https://user-images.githubusercontent.com/68700549/124289181-0688d800-db20-11eb-8832-9585479cc5a0.PNG" alt="dig10" style="zoom:70%;" />

一般来说，decision tree包含一个根节点，若干个内部节点和若干个叶节点。叶节点对应决策结果，其他每个节点对应一个属性测试，如下图所示

<img src="https://user-images.githubusercontent.com/68700549/124289439-549ddb80-db20-11eb-80d0-4b31b7a9ea4b.png" alt="WeChat Screenshot_20210702102857" style="zoom:50%;" />

决策树的生成是一个递归的过程。在决策树划分的过程中，递归停止的条件

* 当leaf node里所包含的所有样本都是属于同一个类别时，大家都一样了
* 当leaf node里所包含的所有样本特征都一样时，就是没有特征可以继续划分了

### 4.2	划分选择

就是我们选属性的时候，怎么选择最优的划分属性呢？我们希望呢，决策树的分支节点所包含的样本尽可能是属于同一类别的，也就是希望purity越来越高。

#### 4.2.1	Information gain

ID3是以information gain为准则来构建decision tree

那我们怎么来测试这个purity呢？我们可以用information entropy，这个是非常常用的指标。我们要理解，这个entropy计算的是当前节点的purity， how pure is it？假设，当前样本集合$D$中第$k$类样本所占的比例为$p_k(k=1,2,...,|𝒴|)$,$y$表示不同的class labels, 则$D$的information entropy的计算公式就是
$$
Ent(D)=-\sum_{k=1}^{|𝒴|}p_klog_2p_k
$$
$Ent(D)$的值越小，则$D$的纯度越高。说这么复杂，举个例子，计算非常简单。假设现在有两个属性,A and B,进行划分，属性A节点可以划分出[0,0,1,1,0,1,1,1],属性B可以划分出[0,0,0,1,0,0,0,0]. 那么，应该选哪个节点划分呢，很明显，应该选B属性节点。我们也可以进行计算information entropy。
$$
Ent(A)=-(\frac{3}{8}log\frac{3}{8}+\frac{5}{8}log\frac{5}{8})=-(-0.1597382746--0.12757498916)=0.28731326376\\
Ent(B)=-(\frac{7}{8}log\frac{7}{8}+\frac{1}{8}log\frac{1}{8})=-(-0.0507429536-0.11288624837)=0.16362920197\\
$$
很明显，我们要选值较小的那个，因为purity越高。

还有一个要理解的概念，就是information gain，这个就是节点跟分支节点来计算划分的不确定性，一般来说，information gain越大，那么，就意味着使用属性$D$来划分所获得的purity越大。这个$a$就是这个属性上的取值，公式就是
$$
Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
$$
看起来好像很复杂，直接看例子，就非常简单了。假设现有有这么一棵决策树，想看看选某个属性后的划分是怎样，现在要计算每个节点的entropy

<img src="https://user-images.githubusercontent.com/68700549/124296803-5e2b4180-db28-11eb-8b2b-6346fd740d10.png" alt="WeChat Screenshot_20210702112629" style="zoom:67%;" />
$$
Ent(f_1)=0.94,Ent(f_2)=0.81,Ent(f_3)=1\\
$$
紧接着是计算这个划分之后的information gain
$$
Gain(f_1,a)=Ent(f_1)-\sum_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)\\
=Ent(f_1)-\frac{8}{14}H(f_2)-\frac{6}{14}H(f_3)\\
=0.94-\frac{8}{14}\times 0.81 - \frac{6}{14}\times 1\\
=0.0486
$$
 计算完之后，我们也要计算用其他属性划分所得到的information gain，我们选值最大的那个作为划分节点。

#### 4.2.2	Gain_ratio

C4.5是使用gain ratio来作为最优属性划分

Information gain有一个缺点，就是对某一属性，如果这个属性的取值比较多，那么information gain会对这个属性有所偏好，为了减少这个“偏好”带来的不利影响，就有了gain ratio。gain ratio的定义是the ratio of information gain to the intrinsic information. Intrinsic information是那个属性划分过来的。如下图所示，这个红色框框的就是intrinsic information

<img src="https://user-images.githubusercontent.com/68700549/124317728-a311a100-db45-11eb-8c94-b1a19d7d2005.png" alt="WeChat Screenshot_20210702145459" style="zoom:67%;" />

Gain ratio的公式是
$$
Gain_ratio(D,a)=\frac{Gain(D,a)}{IV(a)},\\
\text{where}\\
IV(a)=-\sum_{v=1}^V \frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}
$$
如果属性D的取值a数目越多，那么$IV(a)$就会越大。gain ratio对属性数目取值较少的会有所偏好。C4.5算法的思想是先找出information gain高于平均水平的属性，再从中选择gain ratio最高的。

#### 4.3.3	Gini index

CART decision tree就是用Gini Index来选择划分属性的。数据集$D$的purity，就可以用Gini index来衡量，$y$表示不同的class labels公式是
$$
Gini(D)=\sum_{k=1}^{|𝓎|}\sum_{k'\ne k}p_kp_{k'}\\
=1-\sum_{k=1}^{|𝓎|}p_k^2
$$
Gini index越小，则数据集$D$的purity越高.这里要理解，是对某一个点，那个数据集的purity的计算。举例，下面这个点，$f_1$的Gini index是多少？

$Gini(f_1)=1-(\frac{9}{14})^2-(\frac{5}{14})^2=0.459$

<img src="https://user-images.githubusercontent.com/68700549/124296803-5e2b4180-db28-11eb-8b2b-6346fd740d10.png" alt="WeChat Screenshot_20210702112629" style="zoom:67%;" />

当然了，也有对属性的Gini index,公式就是
$$
Gini\_index(D,a)=\sum_{v=1}^V\frac{|D^v|}{|D|}Gini(D^v)
$$
举例，仍旧是上面那幅图，算属性$f_1$的Gini index
$$
Gini(f_2)=1-(\frac{6}{8})^2-(\frac{2}{8})^2=1-0.5625-0.0615=0.375\\
Gini(f_3)=1-(\frac{3}{6})^2-(\frac{3}{6})^2=1-0.25-0.25=0.5\\
Gini\_index(f_1,a)=\frac{8}{14}Gini(f_2)+\frac{6}{14}Gini(f_3)=0.214+0.214=0.428
$$
我们也仍旧要去算其他属性的Gini index，最后我们Gini index最小的那个属性作为最优划分属性，$a_* = \mathop{argmin}\limits_{a\in A} Gini\_index(D,a)$

### 4.3	Pruning

#### 4.3.1	Pre-pruning

#### 4.3.2	Post-prun

### 4.4	连续与缺失值

#### 4.4.1	连续值处理

#### 4.4.2	缺失值处理

### 4.5	多变量决策树

### 4.6	三个Decision tree algorithms

#### 4.6.1	ID3

以information gain为准则来选择进行划分属性

#### 4.6.2	C4.5

以gain ratio为准则来选择进行划分属性

#### 4.6.3	CART

以Gini index为准则来选择进行划分属性



Partial Dependence Plots: Show by how much the average regression value change with a  feature

## Unit 5	Gradient Boosting

https://jozeelin.github.io/2019/07/19/XGBoost/

### 5.1	GBDT

GBDT的主要优点有

* 可以灵活处理各种类型的数据，包括连续值和离散值
* 在相对较少的调参时间情况下，预测的准确率也可以比较高，这个是相对SVM来说的
* 使用一些健壮的损失函数，对outliers的robustness比较强，比如Huber loss function and quantile loss function

GBDT额缺点

* 由于weak learner之间存在依赖关系，难以并行训练。不过可以通过自采样的SGBT来达到部分并行

#### 5.1.1 Boosting tree

了解GBDT之前，先了解boosting tree,这个跟Adaboost是很不一样的。Adaboost是更新每个样本的weight来去让model去学习，最后每个model也有不同的权重加起来。boosting tree其实就是采用additive model和前向分步算法，以决策树(CART为主)为基函数的boosting method称为boosting tree。Boosting tree是基于residual的一种方式，最后也是加起来作为一个总的model。

对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。boosting tree可以表示为决策树的加法模型,也就是下面这个公式，其中$T(x;\Theta_m)$表示决策树，$\Theta_m$为决策树的参数，M为树的个数
$$
f_M(x)=\sum_{m=1}^MT(x;\Theta_m)
$$
Boosting tree采用前向分步算法(additive manner)。Boosting tree是不用求gradient的，仅仅采用当前的残差，仅此而已。

##### 5.1.1.1 Boosting tree例子

现在我们来看一个例子，现在我们有数据，X=[年龄，工作年限]，y=[收入]，$y_{pred}=$[预测收入]。 

| 年龄 | 工作年限 | 收入(K) | 预测收入 | 残差 |
| ---- | -------- | ------- | -------- | ---- |
| 20   | 2        | 10      | 9        | 1    |
| 22   | 3        | 13      | 11       | 2    |
| 25   | 6        | 15      | 10       | 5    |
| 24   | 2        | 13      | 11       | 2    |
| 28   | 3        | 18      | 12       | 6    |
| 23   | 2        | 12      | 12       | 0    |
| 25   | 5        | 16      | 18       | -2   |

那为什么要用GBDT呢？

如果我们的loss function都是MSE或者exponential loss function，那么每一步优化都是非常简单的，但是对于一般loss function而言，往往每一步的优化都不容易。因此，对于这个问题，就有了GBDT。这是利用最速下降法的近似方法，其关键是利用损失函数的负梯度在当前模型的值$-[\frac{\partial L(y,f(x_i))}{\part f(x_i)}]_{f(x)=f_{m-1}(x)}$作为回归问题提升树算法中的残差近似值，拟合一个回归树，

负梯度就是gradient descent下降最快的方向

#### 5.1.1	例子过程

GBDT算法过程

* step1:初始化，估计使损失韩式极小化的常数值，它是只有一个根节点的树
* step2
  * a): 计算算是函数的负梯度在当前模型的值，将它作为残差的估计，对于平方损失函数，它就是通常所说的残差；对于一般损失函数，它就是残差的近似值
  * b): 估计回归树叶节点区域，以拟合残差的近似值
  * c):  利用线性搜索估计叶节点区域的值，使损失函数极小化
  * d): 更新回归树
* step3: 得到输出的最终模型$\hat{f}(x)$

https://zhuanlan.zhihu.com/p/89572181

https://zhuanlan.zhihu.com/p/57814935

#### 5.1.2	目标函数的构建

如果是做regression，则用MSE作为loss，如果是classification，则用cross entropy loss

##### 5.1.2.1 Cross entropy loss

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

Cross entropy loss formula, M mean the number of classes
$$
loss_{M=2}=-(ylog(p))+(1-y)log((1-p))\\
loss_{M>2}=-\sum_{c=1}^M y_{i,c}log(p_{i,c})
$$

<img src="https://user-images.githubusercontent.com/68700549/124370889-f9680800-dc49-11eb-8841-e82fc5c2487f.png" alt="WeChat Screenshot_20210703215927" style="zoom:50%;" />



##### 5.1.2.2	Additive  training

##### 5.1.2.3	使用Taylor series近似目标函数

##### 5.1.5	重新定义一棵树

##### 5.1.6	如何寻找树的形状 

### 5.2	xgboost

### 5.3	LightGBM

### 5.4	Catboost

#### 





##	Unit 6	SVM

### 6.1	间隔与支持向量

就是有一堆数据，假设$y_i\in \{-1,1\}$. 我们的想法就会是在这堆数据里面找出一个超平面，hyperplane，来把这个数据给划分开，但是嘞，会有比较多超平面，哪个是最好的呢？看下面这张图，我们能看到非常多的线来划分这个两组数据，我们现在要找最好的。

<img src="https://user-images.githubusercontent.com/68700549/123185714-8b129100-d464-11eb-99b7-69ce80676da5.png" alt="1_06GSco3ItM3gwW2scY6Tmg" style="zoom:50%;" />

我们看上图的右边，我们会认为最中间的那条线会好一点，因为它对数据的容忍度会高一点，我们的数据集本身就是很乱的，会有噪音啊，假设有一个新数据进来，其他的hyperplane可能会出现问题，但是中间的容忍度会高，更加robust，相对来说泛化能力会强。

于是，我们就对划分hyperplane，可以用下面那个线性方程来描述
$$
w^Tx+b=0
$$
这里呢$w=(w_1;w_2;...;w_d)$为法向量，就是决定hyperplane 的方向，而b呢，就表示位移项，决定hyperplane与原点之间的距离。所以，hyperplane是可以被$w,b$ 来决定的。

现在就可以计算出每个点到hyperplane之间的距离了，公式就跟点到线的距离一样。任意点到hyperplane的距离为
$$
r=\frac{|w^T+b|}{||w||}
$$
我们再看下面这张图，假设我们hyperplane可以分类正确，那么对于任意点，如果$y_i=1$,可以得到$w^Tx_i+b>0$,如果$y_i=-1$,可以得到$w^Tx_i+b<0$

<img src="https://user-images.githubusercontent.com/68700549/123186629-8cdd5400-d466-11eb-9f12-6ba7dde2a546.png" alt="A-linear-support-vector-machine" style="zoom:50%;" />

我们就可以根据这个公式列出来，
$$
\begin{equation}
w^Tx_i+b\ge +1,\space y_i=+1\\
w^Tx_i+b\le -1,\space y_i=-1

\end{equation}
$$
然后呢，凡是在这两条线上的点，就称为是support vector。那么，这些support到hyperplane的距离也是可计算的。由于呢，我们最远的那两条线是平行的，所以，我们就可以将它们相减，可以得到两条平行线之间的距离，也就是(1)式减(2)式
$$
w^Tx_1+b=1,\space\space (1)\\
w^Tx_2+b=-1,\space\space (2)\\
$$
可以得到
$$
w^T(x_1-x_2)=2
$$
我们再看下面这张图，在得到上面的公式之后，我们要进行向量的点积运算，dot product. 我们可以看到呢，$d_1+d_2$,也就两条平行线之间的距离，是$x_1-x_2$的投影

<img src="https://user-images.githubusercontent.com/68700549/123187718-aaabb880-d468-11eb-9cec-8cd9e1ff8921.png" alt="WeChat Screenshot_20210623211854" style="zoom:50%;" />

也就是说
$$
w^T(x_1-x_2)=||w^T||*||(x_1-x_2)||*cos\theta=2\\
d_1+d_2=||(x_1-x_2)||*cos\theta=\frac{2}{||w||}
$$
这个$\frac{2}{||w||}$也就是margin，我们要最大化，也就是做maximum margin。知道这个公式之后，我们需要去找到$w,b$来maximum这个margin，也就是上面这个值，于是我们可以定义下面这个公式,假设有m个数据，我们总是会有$y_i(w^Tx_i+b)\ge 1$
$$
\mathop{max}\limits_{w,b} \frac{2}{||w||}\\
s.t.\space\space y_i(w^Tx_i+b)\ge 1,\space\space i=1,2,...,m
$$
现在，我们需要做的就是最大化$||w||^{-1}$,但是为了方便运算，可我们可以转化成最小化$||w||^2$. 因为后面要求导，也是为了方便运算，可以加入$\frac{1}{2}$. 于是我们就可以重写上面的公式
$$
\mathop{min}\limits_{w,b} \frac{1}{2}||w||^2,\space\space\space\space\space(6.1式)\\
s.t.\space\space y_i(w^Tx_i+b)\ge 1,\space\space i=1,2,...,m
$$
这也就是SVM的基本型，面试时要答出来，SVM要做的就是最小化这个公式。

### 6.2	对偶问题

我们已经求出要最小化的公式了，我们希望maximize margin来划分这个hyperplane：$f(x)=w^Tx+b$. 根据上面的公式，我们知道得求出$w$才能求出margin距离，那怎么求这个hyperplane呢？ 

方法就是使用Lagrange multipliers(在数学上，Lagrange multipliers的一个作用是优化，求最大值，或者是求最小值。subject to  equality constrains)。Lagrange multipliers的方式是这样的
$$
maximize\space f(x,y)\\
s.t.\space g(x,y)=0\\
\downarrow \\
L(x,y,\lambda)=f(x,y)-\lambda g(x,y)
$$
于是乎呢，我们这里也是照猫画虎，把我们的6.1式进行Lagrange化，我们对每条约束，也就是s.t. 那里，加一个Lagrange multiplier，$\alpha _i\ge 0$. 一个数据会对应一个$\alpha$. 

所以，我们就可以把公式写成, 其中$\alpha_i = (\alpha_1,...,\alpha_m)$.
$$
L(w,b,\alpha)=\frac{1}{2}||w||^2-\sum _{i=1}^m \alpha_i (y_i(w^Tx_i+b)-1),\space\space\space\space\space(6.2式)\\
$$
接下来我们要做的就是求$w,b$,所以，我们要对上面这个公式进行求偏导，得0，我们可以得到

对$w$求偏导
$$
w-\sum_{i=1}^m a_iy_ix_i=0\\
\downarrow \\
w=\sum_{i=1}^m a_iy_ix_i
$$
然后对b求偏导
$$
-\sum _{i=1}^m a_iy_i=0\\
\downarrow\\
\sum _{i=1}^m a_iy_i=0
$$
我们将得到的值代回(6.2式)，就可以得到
$$
L(w,b,\alpha)=\sum_{i=1}^m a_i -\frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^Tx_j
$$
于是乎，我们就可以构造dual问题了也就是
$$
\mathop{max}\limits_{\alpha} \space \sum_{i=1}^m a_i -\frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^Tx_j \\
s.t. \space\space \sum_{i=1}^m a_i y_i=0 \\
\alpha_i\ge 0,\space\space i=1,2,...,m
$$
解出$\alpha$之后，就可以求出$w,b$, 我们可以得到SVM模型为
$$
f(x)=w^Tx+b\\
=(\sum_{i=1}^m \alpha_i y_i x_i)^T x +b\\
=\sum_{i=1}^m \alpha_i y_i x_i^T x +b\\
$$
这里，我们会有一个疑问，每次测试一个新样本，就需要跟每一个数据进行转置乘积并求和，哇，假设有一个亿的数据，那岂不是要有很大的空间来存储。看到这里，我们必须要清楚，我们的6.1式是有不等式约束的，也就是要满足KKT条件
$$
\begin{cases}
\alpha_i\ge0 \\
y_if(x_i)-1\ge0\\
\alpha_i(y_if(x_i)-1)=0

\end{cases}
$$
所以，我们是不用担心这么多数据的，根据KKT条件的第三行，可以得知非support vector的$\alpha=0$. 只有support vector，$y_if(x_i)-1=0$. 所以，这里就有一个SVM的重要性质，就是训练完成后，大部分的训练样本都不需要保留，最终模型只跟support vector相关。

### 6.3	Kernel Function

我们现在的数据集中也许不存在一个能够正确划分样本的超平面，就比如说下图，不能够线性划分，需要投射到高维去划分数据。投射之后，样本就可能在这个特征空间内线性可分。如果原始空间是有限维，就是feature数有限，那么是一定存在一个高维特征空间使得样本线性可分。

<img src="https://user-images.githubusercontent.com/68700549/123436366-8e109d00-d59c-11eb-8b27-868423c86372.png" alt="WeChat Screenshot_20210625100300" style="zoom:50%;" />

但是投射到高维也是存在一定的缺点，会增大计算量，本来是二维的，投射到了三维甚至更高维。计算量增大。我们也知道，维度越高，需要训练的数据也要越多。

先来看看数学模型。现在我们假设把数据$x$映射到高维中，也就是$\phi(x)$，我们的$x$在高维空间的投射.于是，我们模型就可以表示为
$$
f(x)=w^T\phi(x)+b
$$
还是跟上面一样，我们要最大化margin，就是求
$$
\mathop{min}\limits_{w,b}\frac{1}{2}||w||^2\\
\text{s.t. }y_i(w^T\phi(x_i)+b)\ge1,i=1,2,...,m
$$
然后，我们就可以得到dual 问题
$$
\mathop{max}\limits_{\alpha}\sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_iy_j\phi(x_i)^T\phi(x_j)\\
\text{s.t. }\sum_{i=1}^m\alpha_iy_i=0,\\
\alpha_i\ge0,i=1,2,...,m
$$
但是呢，现在会有一个问题，就是投射到高维空间后可能是真的很高维，这就导致$\phi(x_i)^T\phi(x_j)$,计算非常困难，于是呢，我们就设想有这么一个函数，也就是核函数,下面这个公式的意思就是使用一个核函数，使得$x_i,x_j$在高维空间的内积，也就是$<\phi(x_i),\phi(x_j)>$要等于通过核函数对$x_i,x_j$的计算结果。这个理解非常重要，满足这个条件，就被成为kernel trick
$$
k(x_i,x_j)=<\phi(x_i),\phi(x_j)>=\phi(x_i)^T\phi(x_j)
$$
假设我们有了核函数，我们的dual问题就可以重写为
$$
\mathop{max}\limits_{\alpha}\sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_iy_jk(x_i,x_j)\\
\text{s.t. }\sum_{i=1}^m\alpha_iy_i=0,\\
\alpha_i\ge0,i=1,2,...,m
$$
那么，什么样的函数能成为核函数呢？条件是什么？

在与核函数$k(·,·)$做出运算后的矩阵,$G_{ij}=k(x_i,x_j)$，必须是对称矩阵，而且一定是半正定矩阵(positive semidefinite matrix)。半正定矩阵定义是，对于任意不为0的实列向量$x$,都有$x^TGx\ge0$. 满足这两个条件的就可当做核函数使用。

我们希望我们的数据能在高维空间中线性可分，所以kernel function的选择是至关重要的，这也是影响SVM性能的最大变数。

这里列出常用的几个核函数

| 名称                     | 表达式                                                       | 参数                                                      |
| ------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| linear kernel            | $k(x_i,x_j)=x_i^Tx_j$                                        |                                                           |
| polynomial kernel        | $k(x_i,x_j)=(<x_i,x_j>+c)^d$                                 | $d=1,c=0$时变为linear kernel,$c\ge0$控制低阶项的强度      |
| Gaussian kernel(RBF)     | $k(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{2\sigma^2})$           | $\sigma >0$为Gaussian kernel的width，数据需要standardized |
| Laplacian kernel         | $k(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{\sigma})$              | $\sigma>0$                                                |
| Sigmoid kernel           | $k(x_i,x_j)=tanh(\alpha x_i^Tx_j +\theta)$                   | tanh是双曲正切函数，$\beta >0,\theta >0$                  |
| Cosine similarity kernel | $k(x_i,x_j)=\frac{x_i^Tx_j}{||x_i||||x_j||}$                 | 用于NLP,衡量夹角相似                                      |
| Chi-squared kernel       | $k(x_i,x_j)=1-\sum_{i,j=1}^n\frac{(x_i-x_j)^2}{\sqrt{x_i+x_j}}$ | 用于CV, 衡量概率分布相似性，数据非负且L1-normalized       |

Gaussian Kernel参数的意义

<img src="https://user-images.githubusercontent.com/68700549/123446786-36c3fa00-d5a7-11eb-92a3-0e3de0747de6.png" alt="WeChat Screenshot_20210625111922" style="zoom: 67%;" />

对Gaussian Kernel,我们在使用的时候，要讨论一下$\gamma$. Gaussian kernel: $k(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{2\sigma^2})=exp(-\frac{||x_i-x_j||^2}{\gamma}),\gamma=\frac{1}{2\sigma^2}$ .

Effects of $\gamma$:

<img src="https://user-images.githubusercontent.com/68700549/123666132-f06cd600-d806-11eb-9a03-6d24a1a1ea90.png" alt="WeChat Screenshot_20210628114938" style="zoom:50%;" />

RBF 核函数的参数$\gamma$定义了单个样本的影响波及范围，gamma 比较小的话，其影响较小；gamma 比较大的话，影响范围较大。gamma 越大，支持向量越少，gamma 越小，支持向量越多。支持向量的个数影响训练和预测的速度。gamma的物理意义，大家提到很多的RBF的幅宽，它会影响每个支持向量对应的高斯的作用范围，从而影响泛化性能。我的理解：如果gamma设的太大，$\sigma$会很小，很小的高斯分布长得又高又瘦， 会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让无穷小，则理论上，高斯核的SVM可以拟合任何非线性数据，但容易过拟合)；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。


<img src="https://user-images.githubusercontent.com/68700549/124841329-f7a88800-df5a-11eb-826b-c9f089e47685.jpg" alt="v2-b65393d97412927f7430ea474b19c853_r" style="zoom:67%;" />

* As $\gamma$ increases
  * decision function $z\approx y_i$ when $x=x_i$
    * decision function就是$z=\sum_{i=1}^n y_i K(x_i,x)$
    * sum of bell curves
    * positive when near positive samples
    * negative when near negative samples
    * 然后可以classification，就是给一个新的点，$x$, 进行划分，然后用decision function决定是哪一个类别。classification function是$\hat{y}=sign(z)$.
  * classifier fits training data better
  * classification region more complex
* As a classifier, higher $\gamma$ results in:
  * lower bias error
  * higher variance error

如果使用Sigmoid kernel，那么SVM就相当于没有hidden layer的神经网络。

核函数的三个重要性质：

* 若$k_1,k_2$都是kernel function，则对于任意正数$\gamma_1,\gamma_2$,其线性组合,$\gamma_1 k_1 + \gamma_2 k_2$也是核函数
* 若$k_1,k_2$都是kernel function，则函数的直积，$k_1 \otimes k_2 (x,z)=k_1(x,z)k_2(x,z)$ 也是核函数
* 若$k_1$是kernel function，则对于任意函数$g(x)$, $k(x,z)=g(x)k_1(x,z)g(z)$ 也是核函数

### 6.4	soft margin and 正则化

我们的数据集中，是会有噪音的，造成很难线性完全可分。就像下面的这个例子，会相对来说比较难划分，所以我们可以用soft-margin

<img src="https://user-images.githubusercontent.com/68700549/123352808-1c9a0580-d52e-11eb-8307-70dc6dc28303.png" alt="WeChat Screenshot_20210624205225" style="zoom:50%;" />

soft-margin的意思就是在满足maximum margin的同时，允许一些可出错的情况，但是要是这个可出错的情况尽可能少 。我们加一个slack variable，而且，我们这里采用的是hinge loss，优化公式就变为
$$
\mathop{min}\limits_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^m \xi_i\\
s.t. \space y_i(w^Tx_i+b)\ge 1-\xi_i\\
\xi_i\ge0,i=1,2,...,m
$$
加入 $\xi$ 之后，我们就可以允错一些数据了，原来是要大于等于1，现在值变小了，大于等于$1-\xi$就好了，所以会允错。每一个样本都会对应一个$\xi$.因为我们还是要去求$w,b$,所以要先转化成dual问题，第一步就是Lagrange 化
$$
L(w,b,\alpha,\xi,\mu)=\frac{1}{2}||w||^2+C\sum_{i=1}^m \xi_i-\sum_{i=1}^m \alpha_i(y_i(w^Tx_i+b)+\xi_i-1)-\sum_{i=1}^m \mu_i\xi_i
$$
于是，我们需要对$w,b,\xi$进行求偏导得0，于是，我们就可以得到对偶问题
$$
\mathop{max}\limits_{\alpha} \space \sum_{i=1}^m a_i -\frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^Tx_j \\
s.t. \space\space \sum_{i=1}^m a_i y_i=0 \\
0\le\alpha_i\le C,\space\space i=1,2,...,m
$$
对于上面的公式，我们要知道$y_iy_j$,如果两个数据点属于同一类别会使值增加，否则减小。$x_i^Tx_j$是衡量两个数据点的相似性。再看$\sum_{i=1}^m a_i y_i=0$. 不同数据点的$\alpha$值会不一样，但是，对于不同类别(比如说，+1,-1,两种类别)，则权重一致，也就是属于+1样本的权重和，属于-1样本的权重和，这两个的绝对值会相等，因为要和为0.

对于soft-margin的KKT条件是
$$
\begin{cases}
\alpha_i\ge0,\mu_i\ge0 \\
y_if(x_i)-1+\xi_i\ge0\\
\alpha_i(y_if(x_i)-1+\xi_i)=0\\
\xi_i\ge0,\mu_i\xi_i=0
\end{cases}
$$
同样，soft-margin SVM也是只跟support vectors相关。加上slack variables之后，界限被放宽松了，support vector就不单单是在线上的，还包括一些允错范围内的数据点，也算是support vector.

这里我们要讨论一下这个variable C，

Effects of C: C值越大，margin越小.C 越高，容易过拟合。C 越小，容易欠拟合。

<img src="https://user-images.githubusercontent.com/68700549/123664644-83a50c00-d805-11eb-9a1a-e7cd4805bd62.png" alt="WeChat Screenshot_20210628113916" style="zoom:50%;" />

参数C越大，支持向量的数量越少.当 C 逐渐增大的时候，判定边界也越来越复杂，过拟合的风险越来越大，同时，我们也发现支持向量（白色边框的点）的数量越来越少。这是因为当 C 增大时，对于误差的惩罚增大，判定边界趋向于将每一个点都正确地分类，导致支持向量机的margin越来越窄，从而使得能成为支持向量的点的数量越来越少。

<img src="https://user-images.githubusercontent.com/68700549/124841610-a51b9b80-df5b-11eb-82d3-e10b747eebb5.jpg" alt="v2-285f04e1e50ed9a840e4e7a0f19032c5_r" style="zoom:67%;" />

* variable C:
  * Low C
    * will have large margin
    * allow more violations of margin
    * more support vectors
    * reduce variance
  * large C
    * small margin
    * reduce violations and fewer support vectors
    * highly fit to the data. Low bias, and higher variance
    * more chance to overfit

通常我们有如下图所示的几种loss function

* Zero-one loss
* hinge loss: $l_{hinge}(z)=max(0,1-z)$
* exponential loss: $l_{exp}(z)=exp(-z)$
* logistic loss: $l_{log}(z)=log(1+exp(-z))$



<img width="792" alt="lossfunction" src="https://user-images.githubusercontent.com/68700549/123355139-0b9fc300-d533-11eb-8d59-56fbdc332985.png" style="zoom:50%;" >

但是SVM这里为什么要用hinge loss呢？

使用hinge loss后，仍旧保持了稀疏性。我们先看logistic loss，它在后面是不断趋近于0的，但是不等于0，即使SVM分类正确，仍旧有loss，因为不等于0. SVM在1之后，就全部等于0了，也就是分类正确之后，就把那个loss变为0了，这也会有个好处，就是对outlier不敏感，也就是说，我们看logistic，都是在分类正确的前提下，假设分类得好，loss就小，分类得相对差点，loss就大。而Hinge的话就一视同仁，也是在分类正确的前提下，不管你有多好，loss都是0，这也使得SVM的解具有稀疏性. 而且，在假设分类不正确的情况下，因为我们最后是要求导的，从而update这个$w,b$的，我们看logistic，如果分类不正确，越不正确，loss就越大，在gradient的过程中，稍微调一点，变化就很大。而hinge依旧是一视同仁，loss，分得越差，loss越大，但是求导后都是-1.

<img src="https://user-images.githubusercontent.com/68700549/123664015-f6fa4e00-d804-11eb-91d3-fa7cab702f19.png" alt="WeChat Screenshot_20210628113531" style="zoom:50%;" />

Hinge loss的优点

* Convex function，容易优化
* 在自变量小于0的部分，梯度较小(-1),对错误分类的penalty较小
* 在自变量大于0的部分，值为0.只要对某个数据分类是正确的，即使分类正确的可能性足够高，也不用针对这个数据进一步优化了
* 在自变量为0处不可求导，需要分段进行求导
* 求解最优化时，只有support vectors是确定分界线，而且support vector的个数是远远小于训练数据的个数的

Logistic loss的优势主要在于其输出具有自然的概率意义，就是在给出预测标记的用时也给出了概率，能直接用于multi-classification。但是，不具有稀疏性，且依赖于更多的训练样本，预测开销大

### 6.5	SVM multi-class

我们的数据总是多个class的，而不总是两个class的，SVM有两种方法扩展到支持多个类别

* OVR (one versus rest)
  * 这个什么意思呢，就是对于k个类别，就训练k个SVM，第j 个SVM就用来判断任一条数据是属于类别j还是属于非类别j。预测的时候，就看$w_i^Tx+b_i$的值，看哪个大，就属于那个类别。
  * 就比如下面这张图，展示的就是OVR的方法
  * <img src="https://user-images.githubusercontent.com/68700549/123469067-937fde80-d5c0-11eb-893b-b68ce1f8b56f.png" alt="WeChat Screenshot_20210625142047" style="zoom:50%;" />
* OVO (one versus one)，就是对于k个类别的情况，训练$k*(k-1)/2$个SVM，每一个SVM只用于判读任意条数据是属于k中的特定两个类别。预测的时候呢，就用这$k*(k-1)/2$个SVM做$k*(k-1)/2$次预测，最后使用计票的方式来决定数据是被分类到哪一个类别。
  * 就比如下面这张图，展示的就是OVO
  * <img src="https://user-images.githubusercontent.com/68700549/123469644-4e0fe100-d5c1-11eb-9d97-f413fae61db2.png" alt="WeChat Screenshot_20210625142611" style="zoom:50%;" />5

SVM的本质就是最大化margin，找到最好的分界线，因为是要离数据都要远距离，所以这个思想有效地减少过拟合。kernel trick使得SVM可以应用于非线性可分的数据上。SVM的理论非常完美，还有很多的kernel可以用，但是当数据量特别大时，训练比较慢。

## Unit 7 Bayesian

### 7.1	Probabilistic classification

假设现在有一个基本问题，二分类问题来看，就是假设有两个类$w_1,w_2$。假设现在有某个样本$X$, 这个样本要么$X\in w_1$, 要么$X\in w_2$. 我们现在要求$p(w_1|X),p(w_2|X)$.

分类问题就是，如果$p(w_1|X)>p(w_2|X)$,那么$X\in w_1$. 如果$p(w_1|X)<p(w_2|X)$, 那么$X\in w2$.

根据Bayes‘ theorem：$P(A|B)=\frac{p(B|A)·P(A)}{P(B)}$.

于是，我们可以得到
$$
p(w_1|X)=\frac{p(X|w_1)·P(w_1)}{P(X)}\\
p(w_2|X)=\frac{p(X|w_2)·P(w_2)}{P(X)}
$$
我们要比较哪个大，因为分母相同，所以只需要去比较分子的大小。现在的问题就变为，如果$p(X|w_1)·P(w_1)>p(X|w_2)·P(w_2)$,那么$X\in w_1$. 如果$p(X|w_1)·P(w_1)<p(X|w_2)·P(w_2)$, 那么$X\in w2$.

这里呢，$p(w_1),p(w_2)$叫做$w$的先验概率prior probability(要非常关注这个，我们的数据集要尽可能每个类别要先验概率要差不多，也就是数据要均衡，先验概率不一样，对模型会有影响)，就是说有多大的概率是属于$w_1$,有多大的概率是属于$w_2$. 就是$p(w_1)=\frac{\text{the # of } w_1}{\text{total #}}$. $p(w_2)=\frac{\text{the # of } w_2}{\text{total #}}. $ $p(X|w_1),p(X|w_2)$叫做$X$ 在$w$上的条件概率conditional probability。$p(w_1|X),p(w_2|X)$叫做$X$ 在$w$上的后验概率posterior probability。

如果不知道prior probability，则假设所有的prior probability是一样的。在此情况下，分类准则则变为，如果$p(X|w_1)>p(X|w_2)$,那么$X\in w_1$.如果$p(X|w_1)<p(X|w_2)$, 那么$X\in w2$. 

接下来就是去估计$p(X|w)$,或者说，给定一组$\{X_{i}\}_{i=1,...,N}\in w$,如何求$p(X|w)$. 这就是概率密度估计问题。

### 7.2 Naïve Bayesian classifier

限制条件如下：

* $X=[X_1,X_2,...,X_M]$,每个$X_i$的维度的值都是离散的
* $X_i$的每个维度是独立的,就是每个属性独立地对分类结果发生影响

应用，垃圾邮件分类。输入: 一个文件 d. 输出：$d\in C_1$ or $d\in C_2$。就是要么是，要么不是。

训练样本$\{d_i,c_i\}_{i=1,...,N}$. $d=\{w_1,w_2,...,w_p\}$. d的训练样本就是一个一个的单词。我们的model就要去学习$p(d|C_1),\text{and }p(d|C_2)$. 就是已知是垃圾邮件还是非垃圾邮件，然后去训练model。

那么, 我们的垃圾邮件分类问题就变成了$p(d|C)=p(\{w_1,w_2,...,w_p\}|C)$. 这里每一个维度都是离散的。但是呢，单词与单词之间是有语义关联的，不完全独立。但这里假设它独立，而且，对于垃圾邮件的应用，这个假设够用了。
$$
p(d|C)=p(\{w_1,w_2,...,w_p\}|C)\\
=\prod_{i=1}^n p(w_i|C)
$$
我们就是要求$p(w_i|C)$，这个就很简单，数数就够了，这里$v$表示有多少个词汇。
$$
p(w|C_{j_{j\in[1,2]}})=\frac{count(w,c_j)}{\sum_{w\in v}count(w,c_j)}
$$
但是呢，会有一个问题，就是当某个词$w$ 在训练集中有，但是在测试集中是没有的，那么，$p(d|C)=0$. 所以，我们需要做一个小小的改变，就是
$$
p(w|C_{j_{j\in[1,2]}})=\frac{count(w,c_j)+1}{\sum_{w\in v}count(w,c_j)+|v|}
$$
这样就可以保证如果某个单词没有出现，概率变为$\frac{1}{|v|}$.5

所以，*7.1*中提出的问题，如何求$p(X|w)$，按照Naïve Bayesian theory，基于属性条件独立性假设，可以得到 $p(X|w)=\prod_{i=1}^d p(x_i|w)$. 这也就是为什么要要求每个维度的值是离散的，不能是连续的。整体公式可以改写成
$$
p(w|X)=\frac{P(w)}{P(X)}\prod_{i=1}^d p(x_i|w)
$$

### 





## Unit 8	Ensemble learning

### 8.1	个体与集成

我们得先理解什么是ensemble learning，就是通过并结合多个学习器来完成学习任务，有时也被成为multi-classifier system

### 8.2	Boosting

Boosting是可以将一系列的weak learners提升为一个strong learner的方法。看下面这张图

<img src="https://user-images.githubusercontent.com/68700549/124177996-fc62cd00-da7e-11eb-9fe4-381fd923e934.png" alt="WeChat Screenshot_20210701151302" style="zoom:50%;" />

我们在这里先定义wear learner and strong learner。

weak learner是指比random guess稍微好这么一丢丢，比如说random guess的error rate是50%，就一半一半。然后weak learner的error rate是49.999%之类的，就称为wear learner。这个很重要，对理解等会的算法很重要

strong learner是指能得到很好的accuracy。

boosting的过程就是先训练一个weak learner，再根据这个weak learner的表现将数据进行给予权重，分类错误的就权重大点，分类正确的就权重小点。这样会使得这些分类错误的数据在后面能够受到更多的关注。然后再继续训练下一个learner，一直重复下去，每个learner最后也会得到一个权重，之后加权求和。下面这个图就是boosting的过程

<img src="https://user-images.githubusercontent.com/68700549/124179962-83b14000-da81-11eb-8efa-8265a14b6a18.png" alt="WeChat Screenshot_20210701153142" style="zoom: 50%;" />

Steps

* step1: 原始训练集输入，带有原始分布，就是weight是平均的
* step2: 给出训练集中每个样本的权重
* step3: 将改变分布后的训练集输入已知的weak learner，weak learner对每个sample给出预测
* step4: 对此次的weak learner给出权重
* step5：loop到step2，循环达到一定的次数，或者是某度量标准符合要求
* step6: 将weak learner按其相应的权重加权组合形成strong learner

#### 8.2.1	Adaboost

Boosting中最典型的算法就是Adaboost (Adaptive boosting)，下面的就是Adaboost的详细过程

*_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________*

1. Initialize $k$: the number of AdaBoost rounds

2. Initialize $D$: the training dataset, $D=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\},x_i\in X, y_i\in \{-1,+1\}$

3. Initialize weights $w_1(i)=1/n,i=1,...,n,w_1\in \mathbb{R}^n$

4. for r=1 to $k$ do:

5. ​          For all $i$: $w_r(i):=\frac{w_r(i)}{\sum_i w_r(i)}$     [normalize weights]

6. ​          $h_r:=\text{FitWeakLearner}(D,w_r)$

7. ​          $\epsilon _r:=\sum_i w_r(i)  \mathbb{1}(h_r(i)\ne y_i)$     [compute error]

8. ​          if $\epsilon_r >1/2$, then stop

9. ​          $\alpha_r:= \frac{1}{2}log[\frac{(1-\epsilon_r)}{\epsilon_r}]$          [small if error is large and vice versa]

10. $$
    w_{r+1}(i):=w_r(i)\times \begin{cases}
    e^{-\alpha_r}& \text{if }h_r(x_i)=y_i\\
    e^{\alpha_r}& \text{if }h_r(x_i)\ne y_i
    \end{cases}
    $$

    

11. Predict: $h_{final}(x)=argmax_j \sum_r^k \alpha_r \mathbb{1}[h_r(x)=j]$

*_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________*

对上面过程的具体解释：

对于第九行$\alpha$的由来，比较复杂，因为Adaboost的目的要最小化指数损失函数，exponential loss function, $f(x)$ is true label, $H(x)$ is predictive label
$$
l_{exp}(H|D)=\mathbb{E}_{x\sim D}[e^{-f(x)H(x)}]
$$
经过一番求解，各种变化，至于过程不太清楚，可以计算出$\alpha$值，这里记住就好了，用了这个$\alpha$的计算公式，就可以minimize exponential loss function。这个$\alpha$也就表示，如果一个分类器的error rate越大，那么所对应的weight，也就是$\alpha$值，也就越小。如下图所示

<img src="https://user-images.githubusercontent.com/68700549/124397574-996f7100-dcde-11eb-8835-159f181ca473.png" alt="WeChat Screenshot_20210704154318" style="zoom:67%;" />

AdaBoost的训练误差是以指数速率下降的。

然后直接用例子来解释Adaboost

假设我现在有这么一组数据$(X,Y)$, 我们还看到有一行$w_1$, 这个就是每个数据的权重，也就对应上面算法的第三行，初始化，全部都一样，也就是$\frac{1}{N}$. 因为这里有10个数据，所以是0.1

|   X   | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| :---: | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   Y   | 1    | 1    | 1    | -1   | -1   | -1   | 1    | 1    | 1    | -1   |
| $w_1$ | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  |

在这么一组数据上，我们要取误差率最低那一个方法（不管我们用什么model，先弄到error rate最低就行了），这里就不说怎么计算了，在V=2.5的时候，误差率最低(当x=6,7,8的时候分类错误了)。所以，基本分类器就是
$$
G_1(x)=\left\{
\begin{aligned}
1,&& x<2.5 \\
-1 & & x>2.5
\end{aligned}
\right.
$$
这样子，我们看到$G_1(x)$的误差率为$\epsilon_1=P(G_1(x_i)\ne y_i)=0.3$,这个也就是对应上面算法的第七行($\epsilon _r:=\sum_i w_r(i)  \mathbb{1}(h_r(i)\ne y_i)$)。把对应权重加起来。

然后我们计算$G_1$的系数，也就是$\alpha_1=\frac{1}{2}log_2 \frac{1-\epsilon_1}{\epsilon_1}=\frac{1}{2}log_2 \frac{1-0.3}{0.3}=0.6112$. 这个也就是对应上面算法的第九行。此时，我们的model就有了权重，整体model就变为$sign(f_1(x)=0.6112G_1(x))$. 这里我把整体model的分类结果写一下，方便后面对比。此时，我们还是可以看到有3个点是错误分类的，也就是当x=6,7,8的时候分类错误了

|    X     | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| :------: | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|    Y     | 1    | 1    | 1    | -1   | -1   | -1   | 1    | 1    | 1    | -1   |
|  $w_1$   | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  |
| $G_1(x)$ | 1    | 1    | 1    | -1   | -1   | -1   | -1   | -1   | -1   | -1   |
| $f_1(x)$ | 1    | 1    | 1    | -1   | -1   | -1   | -1   | -1   | -1   | -1   |

在然后，我们就update我们每个数据的权重，这个就是对应第十行。我把结果直接填表格里了。update完了之后，我们又会进行新的一轮iteration，我们需要对weight做normalization，也就是对应上面算法的第五行。$w_r(i):=\frac{w_r(i)}{\sum_i w_r(i)}$

我们会看到分类正确的权重变小了，分类错误的权重变大了。

|    X     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|    Y     | 1      | 1      | 1      | -1     | -1     | -1     | 1      | 1      | 1      | -1     |
|  $w_1$   | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| $G_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
| $f_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
|  $w_2$   | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.1976 | 0.1976 | 0.1976 | 0.0582 |

此时，我们已经更新了weights了，update完了之后，我们就要fit另一个weak learner，也同样是去error rate最低。记得，要把更新后的weight也计算进去，也就是对应上面的第六行。这里呢，当V=8.5的时候，error rate最低，所以，基本分类器为
$$
G_2(x)=\left\{
\begin{aligned}
1,&& x<8.5 \\
-1 & & x>8.5
\end{aligned}
\right.
$$
如果用这个分类器，则当x=3,4,5时分类错误。这一轮，$G_2(x)$在这一数据集上的error rate 为$\epsilon_2=P(G_2(x_i)\ne y_1)=0.0582*3=0.1746$.，这个就对应上面算法的第七行。

然后计算$G_2(x)$的系数$\alpha_2=\frac{1}{2}log_2 \frac{1-\epsilon_2}{\epsilon_2}=\frac{1}{2}log_2 \frac{1-0.1746}{0.1746}=1.1205$.

我们就可以得到整体的model $sign(f_2(x)=0.6112G_1(x)+1.1205G_2(x))$. 代进去算就好了,用这个新model代进去算，仍旧可以看到有三个分类错误，也就是当x=3,4,5时。

|    X     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|    Y     | 1      | 1      | 1      | -1     | -1     | -1     | 1      | 1      | 1      | -1     |
|  $w_1$   | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| $G_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
| $f_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
|  $w_2$   | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.1976 | 0.1976 | 0.1976 | 0.0582 |
| $G_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |
| $f_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |

然后我们再次更新weight，在normalization一下，可以得到

|    X     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|    Y     | 1      | 1      | 1      | -1     | -1     | -1     | 1      | 1      | 1      | -1     |
|  $w_1$   | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| $G_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
| $f_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
|  $w_2$   | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.1976 | 0.1976 | 0.1976 | 0.0582 |
| $G_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |
| $f_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |
|  $w_3$   | 0.0236 | 0.0236 | 0.0236 | 0.2218 | 0.2218 | 0.2218 | 0.0801 | 0.0801 | 0.0801 | 0.0236 |

在Iteration一次，仍旧去error rate最低。这里呢，当v=5.5时，error rate最低。所以，基本分类器为
$$
G_3(x)=\left\{
\begin{aligned}
-1,&& x<5.5 \\
1 & & x>5.5
\end{aligned}
\right.
$$
可以得到$G_3(x)$在这个数据集上的error rate为$\epsilon_3=P(G_3(x_i)\ne y_1)=0.0236*4=0.0944$

然后再计算$G_3(x)$的系数$\alpha_3=1.631$.

于是乎，我们现在可以得到新的model：$sign(f_2(x)=0.6112G_1(x)+1.1205G_2(x)+1.631G_3(x))$.

|    X     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|    Y     | 1      | 1      | 1      | -1     | -1     | -1     | 1      | 1      | 1      | -1     |
|  $w_1$   | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| $G_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
| $f_1(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | -1     | -1     | -1     | -1     |
|  $w_2$   | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.0582 | 0.1976 | 0.1976 | 0.1976 | 0.0582 |
| $G_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |
| $f_2(x)$ | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | -1     |
|  $w_3$   | 0.0236 | 0.0236 | 0.0236 | 0.2218 | 0.2218 | 0.2218 | 0.0801 | 0.0801 | 0.0801 | 0.0236 |
| $G_3(x)$ | -1     | -1     | -1     | -1     | -1     | -1     | 1      | 1      | 1      | 1      |
| $f_3(x)$ | 1      | 1      | 1      | -1     | -1     | -1     | 1      | 1      | 1      | -1     |

我们可以看到我们最终的model是可以正确分类的。

这也就能够理解为什么Boosting能够降低bias了，但是Boosting对outliers非常敏感，因为要去尝试fit每一个数据。

那尝试去fit每一个数据，会不会造成overfitting呢？

不会，至于解释，很难解释。据说是有一个generalization error bound，$\epsilon_𝒟 \le\epsilon_D+\tilde{O}(\sqrt{\frac{dT}{m}})$. d是VC dimension of base learners, m is the number of training instances, T is the number of learning rounds, and $\tilde{O}(·)$ is used instead of $O(·)$ to hide logarithmic terms and constant factors. 我们可以看到，数据越多，overfitting的可能性越小，但是当我们使用很多base learner的时候，就是当T很大的时候，有可能会造成overfitting。但是实际上不会，真实中的Adaboost是如下图所示的。

<img src="https://user-images.githubusercontent.com/68700549/124209880-68155c00-dab8-11eb-9831-7b68dff1332d.png" alt="WeChat Screenshot_20210701220459" style="zoom: 50%;" />

按理说，我们应该要按照Occam's razor rule，就是越简单越好。但是Adaboost好像不能按照这个rule。因为你看，test error是要在100 rounds之后才趋于平稳，而train error在第7轮左右就趋于平稳了。而且，我们还看到，好像模型越复杂，test error月底，好像有点违反Occam's razor rule了。现在有一个流行的解释就是margin theory。就是跟SVM一样。意思就是说，模型复杂度增加后，margin可能变大了，the bigger margin, the higher the predictive confidence. 不过关于这一理论，还在研究之中。

还有一个就是Adaboost的training error的upper bound是$exp(-2\sum_{t=1}^T \gamma_t^2),\gamma_t=\frac{1}{2}-\epsilon_t$. 至于证明过程，太复杂



adaboost跟SVM的比较：在实践中，adaboost的泛化性能远没有svm好。因此，在不需要处理大量feature（比如不需要进行特征选择），对运行速度要求不是特别高（非线性svm的使用速度很慢），对模型大小要求不是很高（svm的非线性模型在feature维度高时模型往往很大）等情况下，建议先考虑svm。

### 8.3	Bagging and Random Forest

如果我们想要得到泛化性能强的ensemble，那么ensemble里的每个learner，学习器，就应该要尽可能地相互独立。独立很难做到，我们只是做到尽可能地让学习器之间有较大的差异。给定一个训练集，一种可能的做法就是对训练样本进行采样，产生若干个不同的子集，一个子集对应一个base learner，由于训练数据的不同，base learner之间的差异也会比较大，也就做到了尽可能地相互独立。然而，如果每个子集都完全不同，那么我们learner也就只是学到了一部分的训练数据，也就是迅联地不够。所以呢，bagging就是一个有效的采集数据的方法，从而生成子集。

#### 8.3.1	Bagging

Bagging是并行式集成学习方法最著名的代表。核心思想就是有很多并行的estimators，对预测取平均或者众数来作为结果。下图就展示了bagging思想

<img src="https://user-images.githubusercontent.com/68700549/124177918-e3f2b280-da7e-11eb-9064-905a09450602.png" alt="WeChat Screenshot_20210701151252" style="zoom:50%;" />

操作步骤非常简单，就是把*section2.2.3* 的bootstrap sampling执行T次，得到T个数据子集。下面这张图就很形象地解释了bootstrap sampling。

<img src="https://user-images.githubusercontent.com/68700549/123884847-2cdd2680-d91a-11eb-93d4-ce3364d0eb04.png" alt="WeChat Screenshot_20210629203944" style="zoom: 67%;" />

Bagging的时间复杂度很小，所以run起来非常快，很高效，Bagging之后可以直接用于多分类，回归等问题。

使用bootstrap sampling还给bagging带来一个好处，就是由于每个数据子集大约包含了原数据集的63.2%,还有36.8%是可以作为验证集的来测试泛化性能，这剩下的36.8%的数据就称为out-of-bag estimate。

现在我们来计算一算Bagging的泛化误差，假设$D_i$表示$h_i$实际使用的训练集，令$H^{oob}(x)$表示对样本$x$的包外预测，也就是那36.8%的数据，可以得到
$$
H^{oob}(x)=\mathop{argmax}\limits_{y\in Y} \sum_{i=1}^T \mathbb{I}(h_i(x)=y)·\mathbb{I}(x\notin D_t),
$$
然后就可以计算bagging的泛化误差了
$$
\epsilon^{oob}=\frac{1}{D}\sum_{(x,y)\in D}\mathbb{I}(H^{oob}(x)\neq y)
$$
使用bagging方法之后，是减小了variance，因为bagging到最后是要求平均的，假设每个数据子集是独立同分布，那么，可以得到
$$
Var(\frac{1}{n}\sum_{i=1}^n X_i)=\frac{1}{n^2}Var(\sum_{i=1}^n X_i)=\frac{\sigma ^2}{n}
$$
所以，使用bagging之后，均值方差会是单模型方差的$\frac{1}{n}$.

然而，因为用了bootstrap sampling，所以，数据集并不是独立同分布，这种情况下，假设，单模型之间具有相关系数$0<\rho <1$. 我们可以得到模型均值方差为
$$
Var(\frac{1}{n}\sum_{i=1}^n X_i)=\frac{\sigma^2}{n}+\frac{n-1}{n}\rho \sigma^2
$$
随着n增大，$\frac{\sigma^2}{n}$趋近于0，$\frac{n-1}{n}\rho \sigma^2$趋近于$\rho\sigma^2$. 所以，bagging能够降低整体方差。具体推导去看书。

这里说说boosting和bagging的一个区别

bagging是leverage unstable base learners that are weak because of overfitting.

boosting是leverage stable base learners that are weak because of underfitting.



#### 8.3.2	Random Forest

RF是Bagging的一个主要扩展，就是把很多的decision trees结合一块。这非常简单，RF的步骤是先用Bootstrap的方法先做出额外的数据集，一个model对应一个数据集。然后对于每一个decision tree，都要随机选择一个feature subset(里面有k个features)，假设我们总共有d个features，那么k的推荐取值是$k=log_2d$. 如果k=d，那就是传统的decision tree，如果k=1，那么就是随机选择一个feature进行划分。随机选择k个features的原因是features之间的关联性。

单纯的bagging只是数据不同，因为是用bootstrap来选的数据，还有，bagging也是可以放不同的models的。

对于RF来说，采用的是bagging的思想，然后放一堆的decision trees。每个decision tree的feature选择也不一样。所以，RF主要手两个因素的影响，一个是bootstrap sampling，还有一个是feature selection，这两个因素就使得RF的generalization性能更高。

RF较为稳定，稳定取决于多样性

* 训练样本随机化
* 特征选择是徐计划

Ensemble learning的三个好处

* 从统计方面看，不同model的假设空间不一样，使用单个学习器容易造成误选，泛化能力差，结合后会减少这个风险
* 从计算方面看，降低陷入局部极小的风险
* 从表示方面看，真实假设可能不在学习算法所考虑的假设空间，结合多个学习器，扩大假设空间。



#### 8.4.1	Average



#### 8.4.2	Voting

对于分类任务来说呢，我们会有很多的学习器作为一个集合，我们现在假设有一个数据$x$, 我们要去预测这个数据的分类，假设我们现在有n个学习器，现在把$h_i(x),i=1,2,...,n$作为我们的预测输出，有$\{c_1,c_2,...,c_n\}$。在ensemble learning中，我们要去决定是哪个分类，有多种voting方法

* majority voting

  这种方法就是选哪个类别占的票数超过半数，如果超过半数，就预测为那个标记，如果没有超过，就拒绝预测，拒绝预测怎么办呢？就转到plurality voting。

* plurality voting

  就是预测得票最多的那个标记。我们可能会遇到一种情况，就是有多个类别得到相同且是最多的票，这个时候就随机选一个类别。

  $\hat{y}_f=mode\{h_1(x),h_2(x,...,h_n(x))\}$. 这个mode1表示的就是众数

  <img src="https://user-images.githubusercontent.com/68700549/123847328-7ad73780-d8e4-11eb-8695-b101a80d71c5.png" alt="WeChat Screenshot_20210629141521" style="zoom: 67%;" />

  那为什么plurality voting是work的呢？我们举例说明，我们先用binary classification 来看

  * 假设现在有n个独立的分类模型，每一个错误率，error rate为$\epsilon$,独立意味着误差是不相关的。

  * 假设每一个分类模型的error rate都只是比random guess好一点(error rate < 0.5)

    $\forall \epsilon_i \in \{\epsilon_1, \epsilon_2,...,\epsilon_n\},\epsilon_i<0.5$ 

  * 那么k个模型的输出是错误类别，造成最终是错误类别的概率为
    $$
    P(k)=\left( \begin{array}{c} x \\ y \end{array} \right)\epsilon^k(1-\epsilon)^{n-k},k>[n/2]
    $$
    下面这张图可以看出只要我们的$h_i$的error rate是低于0.5的，majority voting是有效的，减少了犯错的概率。蓝色的是ensemble error，橙色的是base error

    <img src="https://user-images.githubusercontent.com/68700549/123848541-d7872200-d8e5-11eb-9219-f9ada47b2116.png" alt="WeChat Screenshot_20210629142514" style="zoom:50%;" />

* weighted voting

  就是$H(x)=c_{\mathop{argmax}\limits_{j}}\sum_{i=1}^T w_ih_i^j(x)$. 我们对每个学习器加一个权重，比如说有些是被认为分得比较好的，可以给的权重大些，被认为分得较差的，权重给小些。通常$w_i\ge0,\sum_{i=1}^T w_i=1$. 

  这里，我们要知道我们的学习器输出的值可能会不一样，有些是直接输出类别，(1或0) [管这个叫类标记]这样投票叫做hard voting。这个少数服从多数就是hard voting，看下面这张图

  <img src="https://user-images.githubusercontent.com/68700549/123851431-2bdfd100-d8e9-11eb-89a5-9df39082f265.png" alt="WeChat Screenshot_20210629144901" style="zoom:60%;" />

  有些是输出概率的，$p(j=0|x),p(j=1|x)$, [管这个叫类概率]，这样的投票叫做soft voting。看下面这张图，当然了，我们会加善一些权重，比如说下面的例子，$h_1,h_2, h_3$分别对应的权重是0.2,0.2,0.6. 最后我们再选出是哪一个类别。但是这个方法就要求每个模型都要输出类概率。soft voting的结合会比hard voting好

  <img src="https://user-images.githubusercontent.com/68700549/123851600-62b5e700-d8e9-11eb-9c7f-ceceb4274fd8.png" alt="WeChat Screenshot_20210629145033" style="zoom:67%;" />

  如果我们的集合中又有概率，又有标记的，处理方式是可以先试试把类标记用Platt scaling， isotonic regression等方法进行校准，得到类概率，进行soft voting。但是也有情形是类概率不能直接进行比较的，就要把类概率转化成类标记输出，然后再投票，hard voting。

### 8.5	Stacking

这个用的很少，就稍微介绍一下。主要还是Boosting and bagging。思路也非常简单，就是当training set很多的时候，可以与一个meta learner结合在一块。如下图所示，很多的个体学习器，就被称为初级学习器。用于结合的学习器称为meta-learner。初级学习器可以用boosting，也可以用bagging。

![WeChat Screenshot_20210701223807](https://user-images.githubusercontent.com/68700549/124212493-0b687000-dabd-11eb-8a79-1b66b84cae66.png)

过程就是我们把数据放到初级学习器中进行学习，然后用初级学习器的每个输出作为一个新的训练集

<img src="https://user-images.githubusercontent.com/68700549/124213226-30a9ae00-dabe-11eb-95cd-f69838ca33f5.png" alt="WeChat Screenshot_20210701224625" style="zoom:67%;" />

但是呢，这样非常容易overfitting，所以得用cross-validation or hold out，用训练初级学习器没有使用的样本来产生meta learner的training set是比较好的。有研究表明，用类概率作为meta-learner的输入会比较好

## Unit 9	Clustering

Clustering的应用场景有很多，比如说图像分割啊，用户分群啊，行为聚类啊之类的

### 9.1	K Nearest Neighbor

KNN是supervised learning。KNN的一个常见应用是推荐系统

KNN的过程非常简单，步骤就这么几步：

* 计算test sample与每个训练样本的距离，不同的距离公式导致后面的排序选择可能会不一样。所以，选好这个距离公式也很重要
* 按照distance升序排列
* 取前k个最近距离的训练数据
* 取那k个中，频率最高的类别作为test sample的预测类别。

我们来观察一下k的取值对分割界面的影响。可以发现，当k取值比较小的时候，分界线会比较复杂，当k取值较大的时候，分界线趋向平滑。KNN对周边数据非常敏感

<img src="https://user-images.githubusercontent.com/68700549/124853162-714c7000-df73-11eb-97f6-68220840274f.png" alt="WeChat Screenshot_20210707223351" style="zoom: 50%;" />

当k的取值变大，近似误差会变大，模型变得简单，underfitting。当k的取值较小，敏感性增强，模型变得复杂，容易发生过拟合，overfitting。我们一般用cross validation来去k值

优缺点：

* 优点
  * 直观，好理解
  * 局部分布，不需要估算整体
* 缺点
  * 局部估算可能不符合全局分布
  * 不能计算概率
  * 对k的取值敏感

### 9.2	K-means

#### 9.2.1	过程

K-means是 unsupervised clustering algorithm.

过程也很简单

* 初始化：随机选择k个点，作为初始的中心点，每个点代表一个group
* 交替更新：
  * 计算每个点到所有中心点的距离，把最近的距离记录下来，然后就把这个点并入这个group中，时间复杂度为O(kn). 
  * 针对每个group的所有的点，计算平均值，并作为这个group新的中心点。时间复杂度为O(n)

看下图，就可以理解

<img src="https://user-images.githubusercontent.com/68700549/124845772-625ec100-df65-11eb-9008-f053585038b5.png" alt="WeChat Screenshot_20210707205317" style="zoom:67%;" />

关于K-mean的几个问题

* k-mean的目标函数是什么，objective function？

  已经dataset $D=(x_1,x_2,...,x_n)$, 每个数据都是d-维向量。k-means需要把这n个samples划分到k个集合中(k≤n)。使得组合平方和(WCSS, within cluster sum of square)最小。也就是，找到满足下面这个公式的cluster $S_i$,
  $$
  \mathop{argmin}\limits_{s}\sum_{i=1}^k \sum_{x\in S_i} ||x-\mu_i||^2
  $$
  这里$\mu_i$是$S_i$group中的所有点的均值。

* 是否一定会收敛？

  将N个数据分为k个clusters，最多有$k^N$中可能，就是每个数据都在k个cluster中走一遍。对于每次iteration，我们是基于旧的clusters从而产生新的clusters

  * 如果旧的clusters跟新的一样，那么下一次iteration还是会一样的
  * 如果进的clusters跟新的不一样，那么objective function就会变小

  所以，K-means是一定会converge的。但是呢，可能会stuck in a local minima. 我们需要一个好的initiation。

* 不同初始化是否会带来不同的结果？

  是的。最终k-means是会取决于initial condition的。如下图所示

  <img src="https://user-images.githubusercontent.com/68700549/124846619-32182200-df67-11eb-8406-65f75dc44ac6.png" alt="WeChat Screenshot_20210707210620" style="zoom: 80%;" />

  如果initialization选的不好，最终cluster得也会不好。怎么办嘞？解决方法有几个

  * K-means++
  * 观察数据选好initialization。
  * 多来几个k-means，多选几次

* 如何选择k的个数？

  k的个数不同，最终的结果也会不同。如下图所示

  ![WeChat Screenshot_20210707210939](https://user-images.githubusercontent.com/68700549/124846852-a8b51f80-df67-11eb-9cfb-b2945aa2e411.png)

  那我们就用inertia来作为一个标准，意思就是计算出每个group的所有点离其所属cluster中心的距离的总和。再把每个cluster的总距离加起来。如下图所示，当达到拐点，elbow point的时候，k是最佳值。我们需要去试

  ![WeChat Screenshot_20210707212522](https://user-images.githubusercontent.com/68700549/124847920-db601780-df69-11eb-9e1e-ce550d1b108b.png)

K-means有一个应用是图像矢量量化。什么意思来，就是当图片传输的时候，进行压缩。我们用RGB表示颜色的时候，一共会用到24bit color，就是每一个channel是8bit。自行google 24 bit RGB image。也就是我们传输数据的时候，也会把$2^{24}$这么多的数据给传过去。就非常大。如果，我们做一个聚类，因为一张图片并不是用到所有颜色，做完聚类后，如下图，当k=64的时候，跟原图看起来差不多。于是乎我们就可以穿64种RGB color，以及512*512 每个pixel对应的RGB颜色即可。就可以做到压缩图片，传输时也不需要传这么多的数据。

<img src="https://user-images.githubusercontent.com/68700549/125147513-d8942c80-e0f9-11eb-9eaf-31bee53a7096.png" alt="WeChat Screenshot_20210709210746" style="zoom: 50%;" />

#### 9.2.2	k-means++

因为我们的初始化选点会对我们的结果造成很大的影响。所以，就提出了k-means++这个方法来做初始化的带你

过程是

* 从数据上随机选一个点作为中心
* 计算所有数据到最近一个点(假设已经有了多个点)的距离$d(x)$
* 接着是给每个点都加权一个概率，这个概率是跟$d(x)^2$成正比，也就是离中心点越远的点，越大概率被选中作为新的中心点
* 重复上面两个步骤，选出k个点
* 已经初始化好了k个点了，就正常使用k-means了

过程如下图所示，我们看④，这里要计算利最近的那个中心点的距离

<img src="https://user-images.githubusercontent.com/68700549/124849311-81ad1c80-df6c-11eb-95c3-229c9ef6370d.png" alt="WeChat Screenshot_20210707214420" style="zoom: 50%;" />

K-means++的优缺点：

* 缺点
  * 计算量大。
* 优点
  * 可以让k-means converge地更快。
  * improve quality of local optimum
  * 总体缩短了运算时间，因为k-means converge地更快了



#### 9.2.3 Difference between KNN and K-means

kNN

* supervised
* 关注局部样本
* k表示的最近距离的训练数据
* 预测后是知道这个test sample是属于哪个class

K-means

* unsupervisied
* 关注全局样本
* k表示的是整个数据集应该分成几个clusters
* test sample的预测只知道是属于哪个cluster，不知道这个cluster代表什么

### 9.3	Mean Shift

Mean Shift是unsupervised learning。之前的K-means还要选k的值。现在Mean shift是不用选k的值的，完全的无参数

### 9.4	DBScan



DBSCAN的主要优点有：

　　　　1） 可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。

　　　　2） 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。

　　　　3） 聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。

DBSCAN的主要缺点有：

　　　　1）如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。

　　　　2） 如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。

　　　　3） 调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值$\epilon$，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。

## Unit 10	Dimension Reduction降维跟度量学习

### 10.1 Curse of dimensionality

随着维度的增加，model performance会先上升后下降，特征数量超过一定值的时候，分类器的效果反而下降，这就是curse of dimensionality。如下图所示

<img src="https://user-images.githubusercontent.com/68700549/126655838-62cdf38f-8b67-41cf-9fd4-0d0945d9cf91.png" alt="1_y09eem2__rqPydHNeDxwCg" style="zoom:50%;" />

如果维度很大，我们可能会遇到的问题

* 如果features比data instance还多，那么，极有可能会overfitting

* 如果有很多features，那么clustering会非常困难。因为当有很多features的时候，基本上每个数据之间都是equidistant。差不多等距。然后clustering又是要用距离来计算的，这会导致clustering非常困难

  我们来看一下，我们这里先定义两个distance，分别是$distmin(x_i)$ and $distmax(x_i)$，分别表示的意思是Euclidean distance to closest point $x_j$ from $x_i$. Euclidean distance to farthest point $x_k$ from $x_i$​.

  当维度较小的时候
  $$
  \frac{distmax(d)-distmin(d)}{distmin(d)}>0
  $$
  当维度很大的时候
  $$
  \lim_{d\to\infty} \frac{distmax(d)-distmin(d)}{distmin(d)} \rightarrow 0
  $$
  所以，当维度很大的时候，Euclidean Distance就失去意义了

那么，如何避免curse of dimensionality呢？

一般是降维和feature selection

* 相关性分析，Pearson coefficient
* PCA，LDA降维
* L1 regularization。L1正则方法具有稀疏解的特性，因此天然具备特征选择的特性，但是要注意，L1没有选到的特征不代表不重要，原因是两个具有高相关性的特征可能只保留了一个，如果要确定哪个特征重要应再通过L2正则方法交叉检验；
* decision tree啊来评估feature importance

### 10.2	PCA

PCA是最常用的一种降维方法，PCA是一个无监督的问题，目标就是提取最有价值的信息(基于方差)，PCA是用于linear separable的数据。用了PCA之后，数据的物理意义是没有了的，降维之后，仍然保存了所有信息。所以PCA的目标就是要找到坐标轴，然后把原来的数据投射到这个新的坐标轴中，但是要最大限度保存信息，要尽可能把数据分得开，这也就是为什么要方差要最大。PCA就是把原始数据投射到高维，然后找到PCs来代表原始数据。看下面这张图，假设一个二维的数据要投射到一维中，我们就需要去找一维的那个坐标轴，coordinate，我们需要找到一个coordinate能够最小化projection error才行，所以看下图，红色的和紫色的，红色的能够最小化projection error，而紫色的project error就比红色的大，

<img src="https://user-images.githubusercontent.com/68700549/123560465-480c3280-d770-11eb-80a2-61bfd8489850.png" alt="WeChat Screenshot_20210627172941" style="zoom: 33%;" />

所以，PCA的定义是把n-dimension的data project to k-dimension中，就是要去找k vectors $u^{(1)},u^{(2)},...,u^{(k)}$来作为坐标轴，然后project到这里之后要能够minimize the projection error。

再举个例子，看下图，现在有一个三维的数据，想要project到2为，那么k=2,就需要去找$u^{(1)},u^{(2)}$这个坐标轴来投射这个三维的数据

<img src="https://user-images.githubusercontent.com/68700549/123560585-2790a800-d771-11eb-9533-7dd28a9de5c2.png" alt="WeChat Screenshot_20210627175725" style="zoom:50%;" />

还有一点，PCA不是linear regression，虽然说有点像，linear regression呢，是去找x对应的y值，看下图的左边。而PCA是找一个坐标，而且是orthogonal, 也就是垂直的，正交的看下图的右边。Linear regression是有y variable的，而PCA是没有的，是有features数据

<img src="https://user-images.githubusercontent.com/68700549/123560720-ff557900-d771-11eb-9d0b-e8e1abfe9218.png" alt="WeChat Screenshot_20210627180326" style="zoom:50%;" />

PCA也不能乱用，虽然说能reduce dimensions，但是不能防止overfitting，因为PCA是仅仅作用在$X$上的，而没有去考虑$y$的。所以如果要防止overfitting，优先使用Regularization，而不是PCA。而且，我们train数据时，优先在full data上使用，先不要用PCA

#### 10.2.1	求PCs的两种方式

有两种计算PC的方式，一种是eigen-decomposition，另一种是SVD。

先来讲第一种：过程是

* do mean normalization:$\bar{x}=\frac{1}{N}\sum_{i=1}^N x_i$

* compute the covariance matrix:$Q = \frac{1}{N}\sum_{i=1}^N (x_i-\bar{x})(x_i-\bar{x})^T=\frac{1}{N}\tilde{x}^T \tilde{x}$

  这会生成$P*P$的矩阵，p表示有多少个features。 $\tilde{x}$表示的意思是 data matrix with sample mean removed (rows: $\tilde{x_i}=x_i-\bar{x}$). covariance是衡量每两个维度之间是否有关系，2个维度之间的线性相关性有多大。

* 然后对这个covariance matrix做特征分解，分解出来就eigenvectors and eigenvalues

* 因为一个eigenvalue会对应一个eigenvector，我们对eigenvalue进行从大到小排序，然后那个eigenvector就是我们要找坐标轴。

* 然后我们取k个eigenvectors进行运算，这样就可以把n-dimension reduce 到 k-dimension。

但是eigen-decomposition这种方法有一些缺点，一般来说，就是数据$X$的维度会非常高，covariance matrix的计算量会很大，而且，eigen-decomposition的计算，就是计算eigenvalues and eigenvectors的效率也低。

因为SVD与PCA等价，就可以把PCA的问题转为SVD来解，可以避免$X^TX$的甲酸，效率也会高

第二种方式：SVD：

* do mean normalization
* 直接对数据矩阵$X$做SVD运算，可以得到[U, S, V]=SVD($X$).
* 然后这个$U\in \mathbb{R}^{p\times p}$, 假设$X\in \mathbb{R}^{n\times p}$. 这个$U$就是我们要找的坐标轴
* 然后我们取k个eigenvectors进行运算，这样就可以把n-dimension reduce 到 k-dimension。

那怎么求投射后的数据呢，因取top k个嘛，我们就会得到$U_k\in \mathbb{R}^{k\times p}$. $x_{proj}=X_{n\times p}U_k^T$.

#### 10.2.2	Choose PCs

接下来我们就要看k取什么值合适。我们要保证我们的projection error要最小。我们也已经知道我们已经project的数据，就是$x_{proj}=X_{n\times p}U_k^T$. 我们就需要计算average approximation error，也就是average squared projection error公式就是：
$$
\frac{1}{N}\sum_{i=1}^N ||x_i-x_{proj_i}||^2
$$
然后我们还要求Proportion of Variance (PoV), 公式就是
$$
PoV=\frac{\frac{1}{N}\sum_{i=1}^N ||x_i-x_{proj_i}||^2}{\frac{1}{N}\sum_{i=1}^N ||x_i||^2}
$$
我们设定一个threshold，比如说，我们要保留98%的variance，则$PoV\ge98%$. 我们取到那个k为止，这是可以画图的，比如说，看下面的图，这样我们就可以知道k的取值了。

<img src="https://user-images.githubusercontent.com/68700549/123569767-f5913d00-d794-11eb-9e33-b0d598dbd3cc.png" alt="Cumulative-proportion-of-variance-explained" style="zoom:50%;" />

$PoV$还有一种简单的计算方式，分PCA方法和SVD方法

PCA方法：

* Eigen-decomposition之后，有eigenvalues，$\lambda$. 计算方式是
* $PoV=\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^p \lambda_j}$.

SVD方法：

* [U, S, V]=SVD($X$). 这个S是singular values,而且是一个$p\times p$的矩阵。其矩阵是diagonal matrix，也就是从左上到右下的对角线有值，其余为0.
* $PoV=\frac{\sum_{j=1}^k S_{jj}}{\sum_{j=1}^p S_{jj}}$ 

当我们用PCA的时候，会lose some information，但是如果那个PC的variance，一般不会lose太多



#### 10.2.3	Eigenface

#### 10.2.4	Whitening

http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/

### 10.3	Kernel PCA

一般来说，m个数据点在的d<m维空间中是线性不可分的，但他们在d>=m维空间中则是几乎必然线性可分。这也就意味着，如果我们能将m个数据点映射到一个m维空间中，就能很容易地构建一个hyperplane将数据点作任意分类. Kernel PCA的主要思想就是先把低维数据投射到高维，因为高维线性可分，然后再用PCA寻找PCs。KPCA是用于non-linear separable的数据。举个例子：

<img src="https://user-images.githubusercontent.com/68700549/123660303-8aca1b00-d801-11eb-83fc-478cd53171c4.png" alt="2-Figure1-1" style="zoom:67%;" />

假设数据在特征空间的均值为0，就是$\sum_{i=1}^m \phi (x_i)=0,\phi(x_i)\in R^n , x_i\in R^D$.

那么，covariance matrix就是$C=\frac{1}{m}\sum_{i=1}^m \phi(x_i)\phi(x_i)^T$.

得到covariance matrix之后，可以求eigenvectors： $Cv_j=\lambda_j v_j,j=1,...,N$.

为了避免映射到特征空间，使用kernels：$K(x_i, x_k)=\phi(x_i)^T \phi(x_k)$。一般kernel选用RBF。

KPCA的过程

* pick a kernel

* construct the normalized kernel matrix of the data (dimension is $n\times n$), 假设有n个数据点

  $K'=K-1_NK-K1_N+1_NK1_N$. 

  这个$1_N$的意思是N-by-N matrix，而且每个element的value是$\frac{1}{N}$.

* solve  an eigenvalue problem,就是找出$K'$的特征值$\lambda_j$和特征向量$a_j$. 公式：$K'a_j=\lambda_j a_j$

* 对于所有的数据点，都是用这个公式降维: $y_i=\sum_{i=1}^m a_{ji}K(X,X_i),j=1,...,m$

投射到高维之后一般不能用普通PCA的步骤，因为all variations of data are same，出现这个的的原因是kernel scale的错误选择。还有一个缺点是计算量大，因为要计算$K(X,X_i)$. 对于每个数据都要跟整体的数据kernel求和

数学方面比较复杂，用起来是很方便的

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel = 'rbf' , gamma = 15)
X_kpca = kpca.fit_transform(X)
```

KPCA的应用一般在于novelty detection，de-nosing images，举例

<img src="https://user-images.githubusercontent.com/68700549/123661184-5b67de00-d802-11eb-9c0d-ddad0c94fb3b.png" alt="WeChat Screenshot_20210628111650" style="zoom:50%;" />

### 10.4	t-SNE

### 10.5	UMAP



离散化

https://blog.csdn.net/CSDN_SUSAN/article/details/103462990

## Unit 11	Feature selection and sparsity

### 11.1	Feature selection

特征选择主要有两个功能：

* 减少特征数量、降维，使模型泛化能力更强，减少过拟合
* 增强对特征和特征值之间的理解

Feature selection的方法有一下几个

#### 11.1.1	Removing features with low variance

这是最简单的feature selection的方法，假设某特征的特征值只有0和1，并且在所有的输入样本中，95%的实例的该特征取值都是1，那就可以认为这个特征作用不大。如果100%都是1，那这个特征就没有意义了。当特征值都是离散型变量的时候这种方法才能用，如果是连续型变量，就需要将其离散化。而且实际当中，一般不太会有95%以上都取某个值的特征存在，所以这种方法虽然简单但是不太好用。可以把它作为特征选择的预处理，先去掉那些取值变化小的特征，然后再从接下来提到的的特征选择方法中选择合适的进行进一步的特征选择。

#### 11.1.2	Univariate feature selection

单变量特征选择能够对每一个特征进行测试，衡量该特征和相应变量之间的关系，根据得分扔掉不好的特征这种方法比较简单，易于运行，易于理解，通常对于理解数据有较好的效果（但对特征优化、提高泛化能力来说不一定有效）。

这里呢，就有以下几种方法：

* Pearson correlation

  非常简单，常用的方法之一，能帮助理解特征和响应变量之间关系的方法，该方法衡量的是变量之间的线性相关性，结果的取值区间为[-1，1]，-1表示完全的负相关，+1表示完全的正相关，0表示没有线性相关。

  Pearson correlation的一个明显缺陷是，作为特征排序机制，他只对线性关系敏感。如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0。如果仅仅根据相关系数这个值来判断的话，有时候会具有很强的误导性，如Anscombe’s quartet，最好把数据可视化出来，以免得出错误的结论。

  尽管有以下的MIC和距离相关系数在了，但当变量之间的关系接近线性相关的时候，Pearson相关系数仍然是不可替代的。

  * 第一、Pearson相关系数计算速度快，这在处理大规模数据的时候很重要。
  * 第二、Pearson相关系数的取值区间是[-1，1]，而MIC和距离相关系数都是[0，1]。这个特点使得Pearson相关系数能够表征更丰富的关系，符号表示关系的正负，绝对值能够表示强度。当然，Pearson相关性有效的前提是两个变量的变化关系是单调的。

* Mutual information and maximal information coefficient (MIC)

  https://medium.com/@rhondenewint93/on-maximal-information-coefficient-a-modern-approach-for-finding-associations-in-large-data-sets-ba8c36ebb96b

  这个也是可以直接调包使用的

  优点

  缺点

  

* distance correlation

  距离相关系数是为了克服Pearson相关系数的弱点而生的。在x和x^2这个例子中，即便Pearson相关系数是0，我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是0，那么我们就可以说这两个变量是独立的。

* model based ranking 

  这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。其实Pearson相关系数等价于线性回归里的标准化回归系数。假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。

#### 11.1.3	Linear model and regularization

正则化的线性模型对于特征理解和特征选择来说是非常强大的工具。L1正则化能够生成稀疏的模型，对于选择特征子集来说非常有用；相比起L1正则化，L2正则化的表现更加稳定，由于有用的特征往往对应系数非零，因此L2正则化对于数据的理解来说很合适。由于响应变量和特征之间往往是非线性关系，可以采用basis expansion的方式将特征转换到一个更加合适的空间当中，在此基础上再考虑运用简单的线性模型。

#### 11.1.4	Random forest

随机森林具有准确率高、鲁棒性好、易于使用等优点，这使得它成为了目前最流行的机器学习算法之一。随机森林提供了两种特征选择的方法：

* mean decrease impurity
* mean decrease accuracy。



safd

随机森林是一种非常流行的特征选择方法，它易于使用，一般不需要feature engineering、调参等繁琐的步骤，并且很多工具包都提供了平均不纯度下降方法。它的两个主要问题，1是重要的特征有可能得分很低（关联特征问题），2是这种方法对特征变量类别多的特征越有利（偏向问题）。

#### 11.1.5	两种顶层特征选择算法

之所以叫做顶层，是因为他们都是建立在基于模型的特征选择方法基础之上的，例如回归和SVM，在不同的子集上建立模型，然后汇总最终确定特征得分。

* 稳定性选择

  稳定性选择是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。

* 递归特征消除

  递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。

  RFE的稳定性很大程度上取决于在迭代的时候底层用哪种模型。例如，假如RFE([Recursive Feature Elimination](https://machinelearningmastery.com/rfe-feature-selection-in-python/))采用的普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；假如采用的是Ridge，而用Ridge正则化的回归是稳定的，那么RFE就是稳定的。

### 11.4	Regularization

https://www.youtube.com/watch?v=Q81RR3yKn30

https://www.youtube.com/watch?v=NGf0voTMlcs

https://www.youtube.com/watch?v=1dKRdX9bfIo&t=0

Regularization是对ML算法做出一些修改，目的是要减少generalization error，但是不减少training error。

一般的做法就是cost function = Loss + Regularization term。就是在等式中添加Regularization term，通过抑制模型中的系数防止overfitting。Regularization的假设是较小的权重对应更简单的模型。模型的复杂度跟参数向量有关，所以，如果让参数趋近于0或者等于0，模型复杂度就下降了。一般的做法就是加上norm，norm是一种定义向量大小的方法，标准公式是
$$
||x||_p=(|\sum_{i=1}^n x_i|^p)^{\frac{1}{p}}
$$
所以，一般加上Regularization做法就是,加上之后使得结构风险最小化。
$$
J(w)=L(w)+\lambda ||w||_p
$$

一般来说，我们常用的Regularization就是两个，一个是lasso(L1), and ridge (L2).就主要来说说这两个的区别。

L1 norm:

公式是$||w||_1=|w_1|+|w_2|+...+|w_n|$

L1 norm的特点：

* 具有稀疏性，可做feature selection，通常是特征数量巨大时的首选模型
* L1 norm对outliers更具抵抗力，因为有些特征系数变为了0，就影响不大了。把一些变量值对结果的影响变为0，就像删除了一样
* 在零点处连续但是不可导，需要分段求导
* 跟ridge regression相比，通常效果不佳

现在来思考为什么L1 norm具有sparsity 的优点。我们现在以二维来看，也就是只有$w_1,w_2$. 我们现在来先理解等值线，contour line。假设我们现在有一个regression function，就是$z=x^2+y^2$. 如果画出来，就是如下图所示，所谓contour line，就是从上往下看，因为我们要做gradient decent，我们需要找到最优点，其实也就是最底部，把左边的图按contour line来画，就是右边的图。

<img src="https://user-images.githubusercontent.com/68700549/123546314-1a9c9600-d72a-11eb-9603-9d1558d5d530.png" alt="WeChat Screenshot_20210627092844" style="zoom:50%;" /> 

知道contour line之后，我们就把L1-norm给画出来。就如下图所示。因为cost function加入L1-norm之后，也就是penalty，L1-norm的最优点是在原点，也就是方形的中心，而原本的loss function，$L(w)$,的最优点，也是在椭圆的中心。两者中和之后，交点也就是最优点，最优点不会发生在其他地方，这个交点，一定会是在其中一个轴上，这样就会导致一些$w$的取值为0.

<img src="https://user-images.githubusercontent.com/68700549/123546447-867efe80-d72a-11eb-8d4a-3e0c37737aad.png" alt="WeChat Screenshot_20210627093153" style="zoom: 80%;" />

而L2-norm是一个圆，原先的loss function与L2-norm的最优解会发生在圆的线上，交点在轴上的概率会低。这也就是为什么L2-norm不会产生sparsity。

<img src="https://user-images.githubusercontent.com/68700549/123546685-a82cb580-d72b-11eb-9c97-fbc82009befb.png" alt="WeChat Screenshot_20210627093958" style="zoom: 80%;" />

L2-norm特点

* 容易计算，可导，适合基于梯度的方法
* 将一些权值缩小并且接近0
* 相关的预测特征对应的系数值相似(collinear)
* 当特征的数量巨大时，计算量会比较大
* 对于有相关特征存在的情况，会包含所有这些相关的特征，但是相关特征的权值分布取决于相关性
* 对outliers非常敏感
* 相对于L1-norm更加精确

L2的目的是使系数的绝对值减小，对绝对值越大的系数，减小的程度越强。它是使得大多数系数都不为0，但是绝对值都比较小

还有一个可以了解一下的是Elastic net，也就是把L1-norm和L2-norm结合起来的，公式是
$$
L_{Elastic}(w)=\frac{\sum_{i-1}^n (y_i-wx_i)^2}{2n}+\lambda (\frac{1-\alpha}{2}\sum_{j=1}^m \hat{w_j}^2+\alpha \sum_{j=1}^m |\hat{w_j}|)
$$
这里呢，$\alpha$控制ridge and lasso的强度

#### 11.4.1	One Standard Error Rule



## Unit 12	计算学习理论

computational learning theory。这是ML的理论基础。通过分析学习任务的困难本质，为学习算法提供理论保证，根据分析结果指导算法设计。这个理论要回答的问题是

* 在什么样的条件下成功的学习时可能的？
* 在什么样的条件下某个特定的学习算法可保证成功运行？

这里考虑两种框架

* PAC
  * 确定了若干假设类别，判断它们能否从多项式数量的训练样例中学习得到
  * 定义了一个对假设空间复杂度的自然度量，有它可以界定归纳学习所需的训练样例数目
* 出错界限框架
  * 考查了一个学习器在确定正确假设钱可能产生的训练错误数量

我们主要解决的问题是：

* 需要多少训练样例才足以成功地学习到目标函数，就是到收敛，需要多少样本
* 学习器在达到目标前会出多少次错，多少次迭代啊之类的

### 12.1	No Free Lunch (NFL)

如果我们不对特征空间有先验假设，则所有算法的平均表现是一样的。

NFL主要表明的是没有最好的算法，说是好，是因为对其有一个假设，在这个假设的条件下，会perform得比较好

关于ML的一个假设，也是基于我们的经验，我们认为：特征差距小的样本，更有可能是同一类

### 12.2	PAC

Probably Approximately Correct

### 12.4	VC维

这个做一个了解即可

Vapnik-Chervonenkis dimension，就是VC维。这个概念是为了研究学习过程一致收敛的速度和推广性（Generalization performance），由统计学理论定义的有关函数集学习性能的一个重要指标。也就是用来衡量研究对象（数据集与学习模型）**可学习性**的指标。

对于机器学习（数据驱动的学习），首先我们要知道他的要素：**训练集**，**测试集**和**学习算法**。

1. 对于训练集来说，训练集要足够大，才能使结果收敛。
2. 对于测试集来说，测试集要有足够的代表性，不能偏差太大。
3. 对于学习算法来说，要足够复杂，以表达特征（X值）与学习目标（y值）之间的逻辑关系。

先看下VC维的定义：一个假设空间H的VC dimension，是这个H最多能够shatter掉的点的数量，记为dvc(H)。

1. 假设空间可以看作模型的复杂度。
2. shatter翻译成打散，指的是不管数据的分布如何，H都要把它区分开。
3. “这个H最多能够shatter掉的点的数指的是无论数据的分布如何”，就是说，不管数据是怎样分布的，H最多能区分多少个数据。我们可以想像，越是复杂的H能够区分的数据点就越多，VC维也就越大。

举个例子，线性分类器的VC维是3

对于3个数据点，一共有$2^3=8$中可能的分布情况，线性分类器总能够分类正确。比如说，看下面的图

<img src="https://user-images.githubusercontent.com/68700549/123471787-53baf600-d5c4-11eb-88d8-2069aecc445b.png" alt="WeChat Screenshot_20210625144729" style="zoom:50%;" />

但是当有4个数据点时，线性分类器就不能够完全分类了，比如说，下面这张图。所以，我们认为线性分类器的VC维是3

<img src="https://user-images.githubusercontent.com/68700549/123471933-7d741d00-d5c4-11eb-8aba-b922d63f796e.png" alt="WeChat Screenshot_20210625144859" style="zoom:67%;" />

### 12.5 Occam's Razor Theory

如无必要，勿增实体。遵循简单有效的原理。

## Unit 13 Semi-supervised



## Unit 14	Probabilistic Model

### 14.1	HMM

### 14.2	MRF

### 14.3	CRF





## Unit 15 Rule Learning





## Unit 16 Expectation maximization algorithm

### 16.1	Maximum likelihood estimation

简要说明一下即可，因为非常简单

Maximum likelihood的两个前提条件是

* 数据独立同分布
* 设定待估参数，使得要求的东西概率最大

这是参数估计的点估计法。按照变量类型可分为两类

* 离散型随机变量

  设总体$X$是离散型随机变量，概率分布为$P(X=t_i)=p(t_i;\theta),i=1,2,...$. 其中$\theta\in \Theta$为待故参数。

  设$X_1,X_2,...,X_n$是来自总体$X$的样本，$x_1,x_2,...,x_n$是每个样本的样本值，称函数
  $$
  L(\theta)=L(x_1,x_2,...,x_n;\theta)=\prod_{i=1}^n p(x_i;\theta)
  $$
  上面的这个函数就是样本$X_i=[x_1,x_2,...,x_n]$的似然函数.如果$\hat{\theta}\in \Theta$，使得$L(\hat{\theta})=\mathop{max}\limits_{\hat{\theta}\in \Theta}L(\theta)$, 这样的$\hat{\theta}$与$x_1,x_2,...,x_n$有关，记作$\hat{\theta}(x_1,x_2,...,x_n$),称为位置参数$\theta$的最大似然估计值，相应的统计量$\hat{\theta}(X_1,X_2,...,X_n)$称为$\theta$的最大似然估计量。

  看起来好像很复杂，举个例子来讲就非常好理解。

  假设我们在射箭，假设我们命中靶心的概率是p，我们并不知道这个p的值，这是我们要去估计的值。我们其实就是根据观察已知的结果去估计这个p的最大值什么，最有可能是哪一个值。假设我每一组射10次，一共射3组，那么就有数据$X_i,i\in [1,3]$。每一组的数据是$x_i,i \in [1,10]$. 假设现在第一组射中4次，第二组射中5次，第三组射中6次。那么每一组情况发生的概率为，第一组$C_{10}^4 p^4 (1-p)^6$, 第二组$C_{10}^5 p^5 (1-p)^5$, 第三组$C_{10}^6 p^6 (1-p)^4$. 然后我们对这个值进行连乘$\prod$. 可以得到
  $$
  L(p)=C_{10}^4 p^4 (1-p)^6\times C_{10}^5 p^5 (1-p)^5\times C_{10}^6 p^6 (1-p)^4
  $$
  我们要求出这个p的最大值即可。求解过程看后面

* 连续型随机变量

  设总体$X$的概率密度函数$f(x;\theta)$，其中$\theta \in \Theta$为待估参数，设$x_1,x_2,...,x_n$是来自总体$X_i$的样本值，称函数
  $$
  L(\theta)=L(x_1,x_2,...,x_n;\theta)=\prod_{i=1}^n f(x_i;\theta)
  $$
  为样本$X_i$的似然函数。如果$\hat{\theta}\in \Theta$,使得$L(\hat{\theta})=\mathop{max}\limits_{\hat{\theta}\in \Theta}L(\theta)$,这样的$\hat{\theta}$与$x_1,x_2,...,x_n$有关，记作$\hat{\theta}(x_1,x_2,...,x_n$),称为位置参数$\theta$的最大似然估计值，相应的统计量$\hat{\theta}(X_1,X_2,...,X_n)$称为$\theta$的最大似然估计量。也是一样的，非常简单。

求解过程

* 写出似然函数(就是上面定义的两个式子，要么离散型，要么连续型)
* 两边取对数，因为是连乘，取对数的话就可以使得乘法变加法，后面求导容易
* 对$lnL(\theta)$求导并且另其等于0
* 求出的解就是最大似然估计值。如果没有解，那么就是边界值，要么最大值，要么最小值，因为是单调增或者是单调减。

举个例题

设总体$X$的概率密度为
$$
f(x;\theta)=\left\{
\begin{array}{rcl}
\frac{\theta^2}{x^3}^{e-\frac{\theta}{x}}       &      & {x>0}\\
0     &      & \text{其他}\\
\end{array} \right.
$$
其中$\theta$为未知参数且大于0，$X_1,X_2,...,X_n$为来自总体$X$的简单随机样本，求$\theta$的最大似然估计量。
$$
L(\theta)=\prod_{i=1}^n f(x_i;\theta)=\frac{\theta^2}{x_1^3}^{e-\frac{\theta}{x_1}}\times \frac{\theta^2}{x_2^3}^{e-\frac{\theta}{x_2}}  ...\frac{\theta^2}{x_n^3}^{e-\frac{\theta}{x_n}} \\
=\frac{\theta^{2n}}{\prod_{i=1}^n}e^{-\theta \sum_{i=1}^n \frac{1}{x_i}}\\
$$
然后我们取对数，可以得到
$$
lnL(\theta)=2nln\theta-ln\prod_{i=1}^n x_i^3 -\theta \sum_{i=1}^n \frac{1}{x_i}
$$
然后求导，可以得到
$$
\frac{dln(\theta)}{d\theta}=\frac{2n}{\theta}-\sum_{i=1}^n \frac{1}{x_i}=0\\
\hat{\theta}=\frac{2n}{\sum_{i=1}^n \frac{1}{x_i}}
$$
就求出来了。

### 16.2	EM algorithm 思路与推演

这个EM算法呢，在许多computational biology 领域有许多应用，EM包含了probabilistic models？我们需要了解这个有什么好，还有就是怎么work的？

probabilistic model，比如说，hidden Markov models，或者是Bayesian networks。probabilistic models在biological data中有广泛的应用。probabilistic models这么受欢迎的主要原因就是从observations中学习这个parameters，很高效，robust procedures。然而呢，可用于训练probabilistic model的data是经常不完整的。举个例子来说，missing values。在医学诊断领域，一个病人的病史通常要包括所有的test 结果，但是真实中这个数据时有限的，会有缺失。这个EM算法可以在probabilistic models中去估计这个参数值。

#### A coin flipping experiment

举个例子来说，coin flipping experiment。我们有两个硬币A, and B。但是我们不知道它们的biases，我们假设是$\theta_A, \theta_B$. 这个意思就是说每次投掷硬币，coin A是head的概率是$\theta_A$, 那么是tail的概率就是$1-\theta_A$. Coin B也是一样，是head的概率是$\theta_B$, 是tail的概率是$1-\theta_B$. 我们的目标是，评估重复下面的过程5次的$\theta$, $\theta=(\theta_A, \theta_B)$. 过程就是

* 随机选取两个硬币中的一个（两者有相同概率）
* 选中之后就要投掷十次。
* 重复上面的步骤5次。因为每次都要投10次，所以总共会投50次。

在实验过程中呢，假定我们一直keep track of $x=(x_1,x_2,...,x_5)，z=(z_1,z_2,...,z_5)$, 这里呢，$x_i\in \{0,1,...,10\}$，表示的在第$i^{th}$轮(这里一共有5轮)的投掷中是head的次数。$z_i\in\{A,B\}$表示的是在第$i^{th}$轮中选到的硬币是哪个。如下图所示，就是这个过程

<img src="https://user-images.githubusercontent.com/68700549/124955030-b492f700-dfe4-11eb-849e-44d518b1071e.png" alt="WeChat Screenshot_20210708120433" style="zoom:67%;" />

parameter estimation，参数估计，在这里的设定中呢，是叫做complete data case，因为在这个model中，所有相关的random variables (每次投掷的结果，当前轮使用的是哪个硬币)都是已经知道的。

这里呢，一个简单的方法来估计$\theta_A,\theta_B$的返回值就是观察到对应硬币是head的次数。上面那张图已经计算出来了。
$$
\hat{\theta}_A = \frac{\text{# of heads using coin A}}{\text{total # of flips using coin A} }
$$
and
$$
\hat{\theta}_B = \frac{\text{# of heads using coin B}}{\text{total # of flips using coin B} }
$$
这个直观的猜测就是maximum likelihood estimation(简单来说，maximum likelihood method就是基于数据在概率上的分布来评估一个统计模型的质量)。maximum likelihood estimation，说的是已知某个随机样本满足某种概率分布，但是其中具体的参数不清楚，参数估计就是通过若干次试验，观察其结果，利用结果推出参数的大概值。如果$logP(x,z;\theta)$是基于已经观察到到的head counts $x$ 和 coin types $z$ 的的联合概率(joint probability(or log-likelihood))的对数, 那么上面的两个公式就可以得到$\hat{\theta}=(\hat{\theta_A}, \hat{\theta_B})$,得到之后就可以最大化$logP(x,z;\theta)$.

现在呢，再想一个parameter estimation变化更为challenging的问题，就是我们现在仅仅知道head counts $x$的数据，不知道$z$,也就是不知道是哪个硬币。我们就认为这个$z$是hidden variables or latent factors。在这种情况下，parameter estimation就是incomplete data case. 这个问题呢，计算每个coin 投掷后是head的比例就不可能知道，因为我们不知道是哪个硬币投的。然而呢，如果我们有某种方法可以填补缺失的数据(在我们投硬币的case中，就是要准确才出每一轮使用的硬币是哪一个)，然后我们就可以在这个问题上减少parameter estimation，因为从incomplete到complete了。

一个用iteration的方法可以得到complete data，按我们这个投硬币的case来说，过程就是，

* 从已知的parameters来决定每一轮投的硬币是A还是B(就是用已知的参数来猜测哪个硬币会投出这个结果,每组投10次嘛，就看这十次的结果和已知参数进行猜测)，可以得到$\hat{\theta}=(\hat{\theta_A}^{(t)}, \hat{\theta_B}^{(t)})$
* 假设猜测正确，那么，我们就在逐步的填补数据，也就是$z$. 
* 采用正常的maximum likelihood estimation的方法来得到$\hat{\theta}^{t+1}$.
* 重复以上步骤直至converge。

因为每一步estimated model都在提升，所以每次填补数据的质量也会越来越好。

然后，expectation maximization algorithm是基于maximum likelihood这个idea的一个提升方法。Maximum likelihood是选取最有可能的那一个coin来作为填补数据，每一轮都是。而expectation maximumization algorithm是用现有的parameters $\hat{\theta}^{(t)}$,来计算每一种可以用来填补数据的概率(这里就有两个，coin A还是coin B)。我们用这个概率来制造一个weighted training set，这个training set就包含了所有可能用来填补的数据，只不过是有weight的。最终呢，a modified version of maximum likelihood estimation就可以来处理这个有weight的training examples，然后就可以得到新的parameter estimates $\hat{\theta}^{(t+1)}$. 使用这个有weighted的samples，而不是纯粹地选一个最好的猜测，这样，expectation maximization algorithm在每轮填补数据后都可以更好地解释model的confidence level。如下图所示

<img src="https://user-images.githubusercontent.com/68700549/125013425-ce125e00-e039-11eb-8b47-c7abde69ea7f.png" alt="WeChat Screenshot_20210708221345" style="zoom:67%;" />

总结，the expectation maximization algorithm就是在两步中不断地循环，第一步就是基于现有的model，也就是parameters，猜测每一种missing data的概率分布(E-step).第二步就是使用刚得到的completion dara来重新估计这个model parameters(M-step).

E-step：不需要完整的数据才能造出这个probabiliy distribution，仅仅需要基于已知数据计算expectation就行，计算期望值。

M-step：就是重新估计的parameters要最大化(maximumization)这个期望值。

#### Mathematical foundations

那么，EM算法是怎么work的呢？为什么必须要这么做？

这个EM算法是maximum likelihood estimation的一种自然产生，因为incomplete data case。EM算法的目的就是从已经观察到的数据中要找到$\hat{\theta}$，可以最大化log probability $logP(x;\theta)$。 普遍来讲，被EM算法解决的优化问题往往会比用maximum likelihood estimation解决的优化问题要困难。因为在complete data case，我们的objective function $logP(x,z;\theta)$往往是可以找到单个单个全局最优，has closed-form solution，就是能找到有限解。而在incomplete data case，往往是有很多local maxima，no closed-form solution，无限解。

为了解决这个问题，EM算法就把优化$logP(x;\theta)$的问题分成了几个简单的子问题来进行优化。这些子问题都是有global maxima的且是closed-form solution。这些对应的子问题的选择就是要保证它们对应的solution($\hat{\theta}^{(1)},\hat{\theta}^{(2)},...$)要能converge到local optimum of $logP(x;\theta)$.

具体来说，因为EM算法在两步中进行循环。在E-step的时候，EM要选择一个function $g_t$, 这个function是$logP(x;\theta)$的下界，其中呢， $g_t(\hat{\theta}^{(t)})=logP(x;\hat{\theta}^{(t)})$， 就是在$\hat{\theta}^{(t)}$要相等. 在M-step的时候，EM算法需要更新这个parameter set $\hat{\theta}^{(t+1)}$使得可以最大化$g_t$. 当下界函数$g_t$与objective function at $\hat{\theta}^{(t)}$相等的时候，会满足如下条件, 因为每次都最大化，所以是小于等于
$$
g_t(\hat{\theta}^{(t)})=logP(x;\hat{\theta}^{(t)})\le g_t(\hat{\theta}^{(t+1)})=logP(x;\hat{\theta}^{(t+1)})
$$
这样子的话，我们的objective function在EM算法的迭代中就是单调递增的，证明过程比较复杂，我们知道就行。绿线线是lower bound，蓝线是log-likehood。

<img src="https://user-images.githubusercontent.com/68700549/125325755-4d8c7f80-e30f-11eb-8514-a85a15c3ab5f.png" alt="WeChat Screenshot_20210712124714" style="zoom:50%;" />

E-step使得lower bound 更加贴紧objective function，M-step最大化lower bound从而使得objective function increase。

对于大部分对nonconcave function的优化方法，EM算法呢，是可以保证收敛到一个local maximum of objective function的。在跑EM过程的时候，使用多个已有值的starting parameters是比较有帮助的。同样，以某种方式来初始化parameters对打破模型的对称性也是非常重要的。

用这些有限的tricks方法，EM算法提供了一个简单且robust的方法来estimates parameters with incomplete data。理论上来说，其他的一些优化方法也是也可以的，比如说gradient descent，Newton-Raphson。但是呢，现实中，EM算法是最简单的而且有效且robust。

### 16.3	EM algorithm 收敛性证明

#### 16.3.1	Jensen's inequality

它的定义是,对于一个continuous concave function $f()$,注意，这里不是convex function啊, 设定概率分布权重$\lambda_j\ge0$, 这个概率权重有$\sum_j\lambda_j =1$. 下面的方程就是Jensen‘s inequality
$$
f(\sum_j \lambda_j x_j)\ge \sum_j \lambda_j f(x_j)
$$
如下图所示

<img src="https://user-images.githubusercontent.com/68700549/125371175-31f49980-e34e-11eb-8618-14b44edcb7de.png" alt="WeChat Screenshot_20210712201721" style="zoom:50%;" />

#### 16.3.2	收敛性证明

input：$\{X_i\}_{i=1,...,N}$

定义：$\{Z_i\}_{i=1,...,N}$

目标，最大化 $E(\theta)=\sum_{i=1}^Nlog(p(X_i|\theta))=\sum_{i=1}^Nlog(\sum_{Z_i}p(X_i,Z_i|\theta))$. 最大似然估计

设$Q_i(Z_i)$为$Z_i$的概率分布，$\sum_{Z_i}Q_i(Z_i)=1$

则可以得到$E(\theta)=\sum_{i=1}^Nlog(\sum_{Z_i}Q_i(Z_i)\frac{p(X_i,Z_i|\theta))}{Q_i(Z_i)}$, 乘以$Q_i(Z_i)$,又除以$Q_i(Z_i)$, 所以跟上面的式子是等价的

根据Jensen’s inequality。因为log是continuous concave function，所以有
$$
E(\theta)\ge \sum_{i=1}^N\sum_{Z_i}Q_i(Z_i)log(\frac{p(X_i,Z_i|\theta))}{Q_i(Z_i)}
$$
这里，必须理解的是log就是那个函数$f()$, 而$Q_i(Z_i)$就是概率分布，这里是先验概率，对Z类别的先验概率

当且仅当$Q_i(Z_i)$与$p(X_i,Z_i|\theta)$成比例时，等号成立

当$Q_i(Z_i)=\frac{p(X_i,Z_i|\theta)}{\sum_{Z_i}p(X_i,Z_i|\theta)}$时，$E(\theta)$取得最大值

这里，把EM algorithm的一般式再写一下

* 随机选$\theta_0$
* E-step
  * $Q_i(Z_i)=\frac{p(X_i,Z_i|\theta_k)}{\sum_{Z_i}p(X_i,Z_i|\theta_k)}$
* M-step
  * 固定$Q_i(Z_i)$, 求$\theta_{k+1}$
  * $\theta_{k+1}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^N \sum_{Z_i}Q_i(Z_i)log(\frac{p(X_i,Z_i|\theta_k)}{Q_i(Z_i)})$
* 重复E-step and M-step直至收敛

 为了证明收敛，就得证明$E(\theta_k)\le E(\theta_{k+1})$

我们假设$M(\theta)=\sum_{i=1}^N\sum_{Z_i}Q_i(Z_i)log(\frac{p(X_i,Z_i|\theta))}{Q_i(Z_i)}$

则E-step做完之后，可以得到$E(\theta_k)=M(\theta_k)$

M-step做完之后，可以得到$M(\theta_{k+1})\ge M(\theta_k)$
$$
E(\theta_{k+1})\ge \sum_{i=1}^N\sum_{Z_i}Q_i(Z_i)log(\frac{p(X_i,Z_i|\theta_{k+1}))}{Q_i(Z_i)}\\
\ge\sum_{i=1}^N\sum_{Z_i}Q_i(Z_i)log(\frac{p(X_i,Z_i|\theta_{k}))}{Q_i(Z_i)}\\
\ge E(\theta_k)
$$
即EM 算法使得似然函数的值单调递增,而且$E(\theta)\le 0$. 小于等于0的原因是，这是log函数，而且，我们是概率密度，所有概率密度加起来小于等于1，也就是在绿色部分，所以有$E(\theta)\le 0$

<img src="https://user-images.githubusercontent.com/68700549/125375103-69674400-e356-11eb-9f69-3dedefb2e371.png" alt="WeChat Screenshot_20210712211615" style="zoom:50%;" />

有上界，且是单调递增的，则一定收敛

### 16.4	EM algorithm and Gaussian mixture model (GMM)

#### 16.4.1	高斯概率密度估计

假设$\{X_i\}_{i=1,...,N}\in C$. 

当$X_i$是一维的情况：
$$
p(X|C)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x-\mu^2}{2\sigma^2}}\\
\mu=\frac{1}{N}\sum_{i=1}^N X_i\\
\sigma^2=\frac{1}{N-1}\sum_{i=1}^N(X_i-\mu)^2
$$
当$X_i$是多维的情况，多维的高斯分布
$$
p(X|C)=\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}e^{-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)}\\
$$
待求参数$\Sigma$，是一个$d\times d$的矩阵.还有$\mu$, 这是一个$d\times 1$的向量.

已知$\{X_i\}_{i=1,...,N}$, 求$\Sigma, \mu$.

构造目标函数（极大似然法，maximum likelihood）
$$
E(\mu,\Sigma)=\sum_{i=1}^N ln p(X_i|C)
$$
这里两个假设：

* 所有$\{X_i\}_{i=1,...,N}$独立同分布（independent and identical distribution）
* 设定$\mu,\Sigma$, 使得出现$\{X_i\}_{i=1,...,N}$概率最大。

$$
E(\mu,\Sigma)=-\frac{Nd}{2}ln(2\pi)-\frac{N}{2}ln\Sigma-\frac{1}{2}\sum_{i=1}^N (X_i-\mu)^T\Sigma^{-1}(X_i-\mu)
$$

然后分别求导另其等于0
$$
\frac{\partial E}{\partial \mu}=0=-\frac{1}{2}\Sigma^{-1}(\sum_{i=1}^N(X_i-\mu))\\
\mu=\frac{1}{N}\sum_{i=1}^N X_i
$$
另外一个求导比较复杂，得出的结果就是$\Sigma=\frac{1}{N}\sum_{i=1}^N(X_i-\mu)(X_i-\mu)^T$. 最后的得出结论就是$\mu$是均值，$\Sigma$是协方差矩阵。

#### 16.4.2	GMM

我们现在已经了解了高斯密度函数估计。我们也知道公式是什么了，这里呢，$\pi$是一个系数，代表一个高斯函数的概率，就是权重
$$
P(X)=\sum_{k=1}^K \pi_k N(X|\mu_k,\Sigma_k)\\
\text{where}\\
N(X|\mu_k,\Sigma_k)=\frac{1}{\sqrt{(2\pi)^d|\Sigma_k|}}e^{-\frac{1}{2}(X_i-\mu_k)^T\Sigma_k^{-1}(X_i-\mu_k)}\\
\sum_{k=1}^K \pi_k = 1
$$
现在呢，用极大似然法去估计概率密度，也就是输入$\{X_i\}_{i=1,2,...,N}$, 就是有N个数据，我们要最小化
$$
E(\{\pi_k,\mu_k,\Sigma_k\}_{k=1,2,...,K})\\
=-\sum_{i=1}^N log[p(X_i)]\\
=-\sum_{i=1}^N log[\sum_{k=1}^K \pi_k\frac{1}{\sqrt{(2\pi)^d|\Sigma_k|}}e^{-\frac{1}{2}(X_i-\mu_k)^T\Sigma_k^{-1}(X_i-\mu_k)}]
$$
这是一个非凸问题，无法求global minima，只能的发哦global minima，可以有几个方法来求解这个方程，有gradient descent啊，heuristic 啊，EM algorithm。

其实EM algorithm就是求local minima的方法，要说上面哪个方法比较好，不一定，因为是local minima，都有可能会比较好。不一定是哪一个。EM algorithm用的也比较局限，只是对某一类局部极值问题可解。EM algorithm的一些优点是

* 不需要调任何参数
* 编程简单
* 理论优美

这里来看看EM algorithm是怎么解GMM方程的

因为$\pi_k,\mu_k,\Sigma_k$是hidden variables，只能根据现有的数据区随机初始化，看先验概率啊啥的

* 随机初始化$\{\pi_k,\mu_k,\Sigma_k\}_{k=1,2,...,K}$,K代表有多少个高斯模型

* E-step

  * soft decision，就是根据概率来判断是属于哪一号高斯模型

  <img src="https://user-images.githubusercontent.com/68700549/125312208-12d01a80-e302-11eb-9ce1-785e4dedb089.png" alt="WeChat Screenshot_20210712111224" style="zoom: 50%;" />

  * 看这个例子，soft decision就是
    $$
    p(X\in \text{1号高斯分布})=\frac{p_1}{p_1+p_2}\\
    p(X\in \text{2号高斯分布})=\frac{p_2}{p_1+p_2}\\
    $$

  * 在E-step，我们要求整体的soft decision，
    $$
    \gamma_{nk}=\frac{\pi_{k}N(X_n|\mu_k,\Sigma_k)}{\sum_{k=1}^K \pi_k N(X_n|\mu_k,\Sigma_k)},\text{where }n=1,...,N \text{ and } k=1,2,...,K
    $$
    一定要理解这个$\gamma_{nk}$的意思，就是说第n个样本落在第k个高斯分布的概率。

* M-step

  * 现在我们可以得到$N_k=\sum_{n=1}^N \gamma_{nk}$，这个的意思就是所有N个样本中，所有属于第k个高斯分布的样本，这个值可能是小数，因为上面求得是soft decision。$\sum_{k=1}^K N_k=N$.

  * 现在，我们要对hidden variables进行更新，记得要去求导更新啊，function已在上面给出。$\pi_k$就是第k个高斯的概率，$\mu_k$就是第k个高斯的均值，$\Sigma_k$是第k个高斯的协方差矩阵
    $$
    \pi_k^{(new)}=\frac{N_k}{N}\\
    \mu_k^{(new)}=\frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}X_n\\
    \Sigma_k^{(new)}=\frac{1}{N_k}\sum_{n=1}^K \gamma_{nk}(X_n-\mu_k^{(new)})(X_n-\mu_k^{(new)})^T
    $$
  
* 重复E-step and M-step，直至收敛。

#### 16.4.3	GMM application



### 16.5	EM与k-means的关系

k-means就是一个简化的EM，一样的

它的问题就是，输入N个样本，$\{X_i\}_{i=1,2,...N}$.,要输出每个样本是属于哪一个类 ,$\{Z_i\}_{i=1,2,...,N}$, 其中，$Z_i=1,2,...,k$.

步骤也是一样的

* 随机初始化$\mu_1,...,\mu_k$

* E-step

  $Z_i=\mathop{argmin}\limits_{k}||X_i-\mu_k||$. 意思就是这个点离谁近，就属于哪一类

* M-step

  $N_k=\sum_{i=1}^NI(Z_i=k)$. 这个表示的意思就是N个样本中有多少个是属于第k类的

  $\mu_k=\frac{1}{N_k}\sum_{i=1,z_i=k}^N X_i$.  这个表示的意思就是$\mu_k$是第k类的所有样本均值

* 重复E-step和M-step直至收敛

$E(\{\mu_k\})=\sum_{k=1}^K \sum_{i=1,z_i=k}^N ||X_i-\mu_k||^2$. 我们要最小化这个方程，这个也就是K-means的loss function。我们知道这里呢，E-step和M-step都会使$E(\{\mu_k\})$变小，而且有下界，就是0，这就说明了这是单调递减且有下界，一定会收敛。



## Unit 17 Convex Optimization

https://www.bilibili.com/video/BV1dt411n7mj?p=4

现在的AI问题，就是模型加优化，所以AI=model+optimization

优化的过程其实就是训练的过程，我们在训练时要用gradient decent，这里就有SGD，Adagrad，Adam，Newton，EM等等，都要去进行适当的了解。

任何一个优化问题，都可以写成如下的形式，这个是标准形式,这里$x$是定义域。如果出现$f_i(x)\ge0$,怎么办？其实就可以转成$-f_i(x)\le0$
$$
Minimize\space f_o(x)\\
s.t.\space f_i(x)\leq 0,i={1,...,k}\\
g_j(x)=0,j=1,...L
$$
下面这张表就是一些需要掌握的loss function，得去总结一下



### 17.1	Optimization Categories

就是我们有了标准化的函数之后，也就是上面所提及的标准化。现在我们需要对其分类，总体来说，是有4类

* Convex V.S non-convex

  * 如果函数是convex，说明是有global minimum
  * 如果函数是non-convex, 说明能找到local minimum，但是不一定能找到global minimum. 一般来说，神经网络的函数就会比较复杂，比较难找到global minimum.所以在DL中，一般会做一些pretraining，用于model parameters的初始化，原因就是要尝试去找global minimum
  * 比如说，下面这张图，就是一个很好的例子
    * <img src="https://user-images.githubusercontent.com/68700549/122807240-52728c00-d299-11eb-9856-4fd295086e64.png" alt="convex-nonconvex" style="zoom: 80%;" />

* Continuous V.S discrete

  * 一般来说，continuous会比较多

* constrained V.S un-constrained

* smooth V.S non-smooth

  * 比如说，L1-regularization 就是non-smooth的
  * ![L1-and-L2-norm-minimization](https://user-images.githubusercontent.com/68700549/122808098-61a60980-d29a-11eb-8257-d069ccb88290.png)

  

### 17.1	Convex Set

首先，我们来看Convex set的定义。假设对于任意$x,y\in C$并且任意参数，$a\in[0,1]$,我们有$ax+(1-a)y\in C$,则集合为convex set.看下面这张图，左边的为non-convex set，右边的为convex set。也就说我们定义的两个点$x,y$,他们之间的连线的所有点，必须在convex function里面的定义域里，那么，那个集合就是convex set

![a-Non-convex-set-b-convex-set](https://user-images.githubusercontent.com/68700549/122808808-56071280-d29b-11eb-8608-1778775bdfc4.png)

Convex Set的一些例子

* 所有的$\mathbb{R}^n$,就是n维的实数的向量
* 所有的正数集合$\mathbb{R}_+^n$
* 范数norm $||x||\le1$
* Affine set,线性方程组的所有解$Ax=b$
* Halfspace:不等式的所有解:$Ax\le b$

答案就是左边的是convex set，任意选择两个点，都会在在里面

中间的和最右边的是non-convex set。最右边的是non-convex的原因是边缘，有部分的边缘是取不到的。所以是non-convex set

![Picture1](https://user-images.githubusercontent.com/68700549/122809199-c6ae2f00-d29b-11eb-9b75-8f44e942a689.png)

还有一些性质就是，两个convex set的交集也是convex

### 17.2	Convex Function

#### 17.2.1	convex function的定义

我们先看convex function的定义

函数的定义域dom$f$为convex set，对于定义域里任意的$x,y$,函数满足
$$
f(\theta x+(1-\theta)y)\le \theta f(x)+(1-\theta)f(y)
$$
如果画出图来就是

<img src="https://user-images.githubusercontent.com/68700549/122810535-78019480-d29d-11eb-86be-b98ab60b59fc.png" alt="Picture2" style="zoom:50%;" />

一样的，都是一样的解释

<img src="https://user-images.githubusercontent.com/68700549/122813675-34a92500-d2a1-11eb-8561-1ef0a6606e29.png" alt="WeChat Screenshot_20210621145840" style="zoom:50%;" />

复杂的函数可以进行拆分成几个简单的函数，凸函数之和也是凸函数。一下是常见的convex function

* 线性函数为convex function。即使是凹函数也没关系，因为凸函数=-凹函数
* $e^x,-logs,xlogx$ 都是凸函数
* 范数都是凸函数
* $\frac{x^Tx}{t}$为convex function ($x > 0$)

#### 17.2.2    凸函数的判定

* First order convexity condition

  假设$f:R^n\rightarrow R$是可导的,differentiable,则$f$为凸函数，当且仅当，if and only if $f(y)\ge f(x)+\bigtriangledown(x)^T(y-x)$, 对于任意$x,y\in domf$. 这个用的比较少

* Second order convexity condition

  假设$f:R^n\rightarrow R$是两次可导的，twice differentiable. 则$f$为凸函数，if and only if $\bigtriangledown ^2 f(x) \succeq 0$, 对于任意$x,y\in domf$.这个会用的比较多。也就是说$f''(x)$ exists everywhere, then $f(x)$ is convex if $f''(x)\ge0$
  
  

### 17.3    Linear Programming

#### 17.3.1	Transportation Problem



#### 17.3.2	Portfolio Optimization



#### 	17.3.3	Set Cover Problem

### 17.4	Duality

#### 17.4.1	Lower bound property

####	17.4.2	Strong and weak duality

####	17.4.3	Complementary slackness

#### 17.4.4	KKT Conditions

### 17.5 Non-linear optimization



## Unit 18	Visualization

synthetic data

AutoML

Permutation Feature Importanc



属于Advanced ML的内容

Bayes rule, Bayesian Inference, maximum likelihood, Exponentiated Gradient, Perceptron, auto-encoders, Empirical Risk Minimization, PAC bounds, Occam's Razor, VC inequality, Structural Risk Minimization, Lipschitzness, Jensen’s inequality, convergence rates, optimization methods (GD, SGD, Netwon, BFGS, LBFGS, ISTA, FISTA, bound majorization), Expectation Maximization (EM), Curse of dimensionality, Johnson–Lindenstrauss lemma, clustering (k-center clustering, k-means clustering, spectral clustering, hierarchical clustering, doubling algorithm, cover tree algorithm, spectral clustering), Gaussian Process, Bayesian Optimization, Markov Chain Monte Carlo (MCMC), variational bayesian methods, Structured SVMs

