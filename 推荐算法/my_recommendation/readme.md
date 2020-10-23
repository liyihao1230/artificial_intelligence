# user_based协同过滤:
	利用pearson相关系数计算用户距离
	pearson/sum(pearson)作为user_based推荐权重

# GBDT/RNN使用编码后特征预测用户评价
	对所有user/movie特征进行编码(one-hot/multi-hot)
	GBDT模型
	DNN模型
	得到评分矩阵(未实现)