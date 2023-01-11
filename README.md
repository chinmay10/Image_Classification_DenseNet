# Image_Classification_DenseNet
Implemented modified DenseNet with multiple dense and
transition layers along with using bottleneck and reduction to achieve image classification accuracy of 94.66% on the public CIFAR-
10 test data.


In this project I present a different approach
to achieve better accuracy using the modified DenseNet with multiple dense and
transition layers along with using bottleneck and reduction. The DenseNet improves
on the Resnet approach where instead of using identity mapping for gradient
propagation, inputs from all of previous layers are concatenated and given to
current layer and it further gives output to all succeeding layers. This approach
has shown to help improve the gradient flow, allowing the network to be compact
and thinner, improving the computational efficiency. Multiple experiments were
done with adding different image augmentations, scheduling, optimizer and using
dropouts to achieve a image classification accuracy of 94.66% on the public CIFAR-
10 test data.
1


Instructions to run
Please ensure the 'data_dir', 'model_dir', 'result_dir' are well defined to the system running.

There are three modes to run, you have to just change the 'mode' in confugre.py file:
1. 'train' - for training model from start
2. 'test' - for evaluating model on public test dataset on the cheeckpoints given in Configure.py file
3. 'predict' - for predicting on the private test dataset

To run on server/please load following modules
module purge
module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load PyTorch/1.9.0
module load tqdm/4.56.2
module load matplotlib/3.3.3


