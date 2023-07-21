# liegroups
Code for [The Lie-Group Bayesian Learning Rule](https://arxiv.org/abs/2303.04397),
E. M. Kiral, T. Möllenhoff, M. E. Khan, AISTATS 2023.

## installation and requirements
The code requires [JAX](https://github.com/google/jax) and various other standard dependencies such as matplotlib and numpy; see the 'requirements.txt'. 

To train on TinyImageNet, you will need to download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 
and extract it into the datasetfolder directory (see the 'data.py' file). 

## examples

### tanh-MLP on MNIST
To run the additive and multiplicative learning-rule proposed in the paper on a tanh-MLP & MNIST dataset, you can use the following commands: 

Running the additive rule:
```
python3 train.py --optim additive --model mlp --alpha1 0.05 --epochs 25 --noise gaussian --noiseconfig 0.001 --batchsize 50 --priorprec 0 --mc 32 --warmup 0
``` 

This should train to around 98%.

Running the multiplicative rule (the code currently only supports Rayleigh-noise):
```
python3 train.py --optim multiplicative --temperature 0.006 --alpha1 50 --beta1 0.9 --model mlp --noise rayleigh --batchsize 50 --mc 32 --epochs 25 --priorprec 0 --warmup 0
```

This should also train to around 98%.

### first layer filter visualizations

To reproduce the figures visualizing the filters, run the following (after training the tanh-MLP networks using the above commands):

```
python3 plot_filters.py --resultsfolder results/mnist_mlp/additive/run_0 
```

![additive filters](https://i.imgur.com/PD3utxC.png)

```
python3 plot_filters.py --resultsfolder results/mnist_mlp/multiplicative/run_0 
```

![multiplicative excitatory](https://i.imgur.com/6v3LWn5.png)
![multiplicative inhibitory](https://i.imgur.com/2NDYJq6.png)

The above filter images are saved by the script in the resultsfolder as png files. 

### CIFAR and TinyImageNet
To run the affine and additive learning rule on CIFAR and TinyImageNet dataset, you can use the following commands: 

Affine update rule (w/ Gaussian noise):
```
python3 train.py --optim affine --temperature 1 --alpha1 1.0 --alpha2 0.05 --beta1 0.8 --beta2 0.999 --dataset cifar10 --model resnet20 --noise gaussian --batchsize 200 --mc 1 --noiseconfig 0.005 --batchsplit 1 --epochs 180 --priorprec 25
```

Running the additive update rule (w/ Gaussian noise):
```
python3 train.py --optim additive --alpha1 0.5 --beta1 0.8 --dataset cifar10 --model resnet20 --noise gaussian --batchsize 200 --mc 1 --noiseconfig 0.005 --batchsplit 1 --priorprec 25
```

To evaluate ECE, nll and accuracy of the trained models, run the following command specifying the folder where the results have been saved:

```
python3 test.py --resultsfolder results/cifar10_resnet20/affine/run_0
```

This produces an output as follows, cf. also Table 2 in the paper:
```
results at g:
  > testacc=91.96%, nll=0.2887, ece=0.0363
results at model average (32 samples):
  > testacc=92.02%, nll=0.2661, ece=0.0247
```

We can also evaluate our additive learning rule:
```
python3 test.py --resultsfolder results/cifar10_resnet20/additive/run_0
```

This produces an output as follows, cf. also Table 2 in the paper:
```
results at g:
  > testacc=92.07%, nll=0.3014, ece=0.0420
results at model average (32 samples):
  > testacc=92.21%, nll=0.2688, ece=0.0268
```

### training with multiple MC samples

We can run the affine learning rule using multiple random samples as follows:
```
python3 train.py --optim affine --temperature 1 --alpha1 1.0 --alpha2 0.05 --beta1 0.8 --beta2 0.999 --dataset cifar10 --model resnet20 --noise gaussian --batchsize 200 --mc 3 --noiseconfig 0.01 --batchsplit 1 --epochs 180 --priorprec 25
```
This will be more computationally expensive but leads to improved results:
```
results at g:
  > testacc=92.13%, nll=0.2747, ece=0.0348
results at model average (32 samples):
  > testacc=92.42%, nll=0.2403, ece=0.0099
```

## troubleshooting

Please contact [Thomas](thomas.moellenhoff@riken.jp) if there are issues or quesitons about the code, or raise an issue here in this github repository.
