# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE



training_configs = {
	'mode':'predict',
	'data_dir':'/../data_CIFAR/',
	'model_dir':'/../saved_models/',
	'result__dir':'/predictions.npy',
	"iter":"iter10",
	"block": 'Bottleneck',
	"num_classes":10,
	'weight_decay':1e-4,
	'learning_rate':0.1,
	"batch_size":64,
	'save_interval':5,
	'max_epoch':200,
	'checkpts': [170,180,190,200]
}

### END CODE HERE
