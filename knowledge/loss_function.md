# 回归模型损失(评价)函数
	目标值离散

## 均方误差(MSE)
	Mean Squared Error
### $$loss = \frac{\sum_{i=1}^m(y_i-\hat{y}_i)^2}{m}$$

#### 含义
	真实值与预测值的差平方求和后求平均
	常用于计算线性回归损失函数

## 均方根误差(RMSE)
	Root Mean Squared Error
### $$loss = \sqrt{\frac{\sum_{i=1}^m(y_i-\hat{y}_i)^2}{m}}$$

#### 含义
	相比较于MSE, RMSE的误差单位与目标值相同, 因此可以更好地描述误差

## 平均绝对误差(MAE)
	Mean Absolute Error
### $$\frac{\sum_{i=1}^m\lvert(y_i-\hat{y}_i)\rvert}{m}$$

#### 含义
	简单粗暴

## R方(R Squared)
### $$R^2 = 1-\frac{\sum_i(\hat{y}_i-y_i)^2}{\sum_i(\hat{y}_i-\overline{y}_i)^2}$$

#### 含义
	分母为模型随机预测的平方误差, 分子为模型的平方误差, 因此$R^2$越大时, 表示模型的效果越好

## 0-1 损失函数
### $$loss = $$

#### 含义
	是非凸函数, 衡量对应分类判断错误个数

## 绝对值、log、平方、指数损失函数
### $$loss = $$

#### 含义
	顾名思义

## Cross-entropy交叉熵损失函数
### $$loss = $$

#### 含义
	本质是对数似然, 可用于二分类和多分类任务