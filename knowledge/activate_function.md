## sigmoid
### $$f(x)=\frac{1}{1+e^{-x}}$$
#### 函数特点
	输入变化为0~1之间的数值, 输出不是zero-centered
	对于神经网络可能导致梯度消失和梯度爆炸
## tanh
### $$tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
#### 函数特点
	输入变化为-1~1之间的数值, 输出是zero-centered
	对于神经网络可能导致梯度消失
## Relu
### $$f(x)=max(0,x)$$
#### 函数特点
	输入为非正数时, 输出为0, 输入为正数时, 斜率为1
	解决了正区间梯度消失的问题
	计算量小, 收敛速度块
	Dead Relu: 有些神经元永远不会被激活
## Leaky Relu/ PRelu
### $$f(x)=max(ax,x)$$
#### As for PRelu, the $a$ is rectified so that it should be written as $a_i$
#### 函数特点
	解决Dead Relu问题
## ELU
### $$f(x)=
\begin{cases}
x,& \text{if x>0}\\
a(e^x-1),& \text{otherwise}
\end{cases}$$
#### 函数特点
	解决Dead Relu问题
	计算量相对大
## MAXOut
#### never used this activate function