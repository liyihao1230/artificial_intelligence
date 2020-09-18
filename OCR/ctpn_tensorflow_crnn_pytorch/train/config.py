class DefaultConfig(object):
	# train_data_root = '/Users/yihaoli/Desktop/cnocr_dataset/new_dataset/data_train.txt'
	# validation_data_root = '/Users/yihaoli/Desktop/cnocr_dataset/new_dataset/data_test.txt'
	# modelpath = './models/pytorch-crnn.pth'
	# image_path = '/Users/yihaoli/Desktop/cnocr_dataset/new_dataset/images'
	train_data_root = '/home/hopson/new_dataset/data_train.txt'
	validation_data_root = '/home/hopson/new_dataset/data_test.txt'
	modelpath = './models/pytorch-crnn.pth'
	image_path = '/home/hopson/new_dataset/images'

	#batch_size = 1
	batch_size = 64
	img_h = 32
	num_workers = 4
	# use_gpu = True
	use_gpu = False
	max_epoch = 1000
	learning_rate = 1e-5
	weight_decay = 1e-4
	printinterval = 20
	valinterval = 100

def parse(self,**kwargs):
	for k,v in kwargs.items():
		setattr(self,k,v)

DefaultConfig.parse = parse
opt = DefaultConfig()