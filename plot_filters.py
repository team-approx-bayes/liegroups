import argparse
import os
import sys
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib

from torchvision.utils import make_grid, save_image

from data import dataloader

def main():
    # options and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testbatchsize', dest='testbatchsize', 
                        type=int, default=200)
    parser.add_argument('--resultsfolder', dest='resultsfolder', type=str, 
                        default='.', required=True)
    
    parser.set_defaults(augment = True)
    args = parser.parse_args()

    with open(os.path.join(args.resultsfolder, 'trainstate.pickle'), 'rb') as file:
        trainstate = pickle.load(file)
        trainargs = pickle.load(file)

    with open(os.path.join(args.resultsfolder, 'info.txt'), 'rt', encoding='utf-8') as file:
        info = file.read()
        print('information of the run that has been loaded:')
        print(info)

    # prepare dataset
    try:
        trainset, testset, trainloader, testloader = \
            dataloader(trainargs.dataset)(trainargs.batchsize, trainargs.testbatchsize, 
                                          trainargs.datasetfolder, trainargs.augment)
        
    except KeyError:
        print(f'Dataset {args.dataset} not implemented.')
        sys.exit()

    ndata = len(trainset)
    ntestdata = len(testset)
    nclasses = len(trainset.classes)

    print(f"""  > dataset={trainargs.dataset} (ntrain={ndata}, """
        f"""ntest={ntestdata}, nclasses={nclasses})""")

    rngkey = jax.random.PRNGKey(trainargs.randomseed)
    np.random.seed(trainargs.randomseed)
    torch.manual_seed(trainargs.randomseed) 
    rngkey, initkey = jax.random.split(rngkey)

    if trainargs.dataset != 'mnist' or ('mlp' not in trainargs.model): 
        print(f'This code is only for MLP/MNIST visualization.')
        sys.exit() 

    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)[0]
    if trainargs.optim == 'additive':
        weights = trainstate.optstate['w']

        act = jnp.abs(jnp.dot(weights['linear']['w'].T, datapoint.reshape(-1, 1)))
        idx = jnp.argsort(-act.ravel())

        filters = jnp.abs(weights['linear']['w'].reshape(1, 28, 28, -1))
        filters = jnp.transpose(filters, [3, 0, 1, 2])
        activated_filters = filters[idx[0:31]]
        activated_filters = jnp.concatenate([datapoint.reshape(1,1,28,28), activated_filters])
        grid = make_grid(torch.from_numpy(np.array(activated_filters)), 
                            nrow=8, normalize=True, scale_each=True, pad_value=1.0, padding=1)
        save_image(grid, os.path.join(args.resultsfolder, f'activated_filters.png'))

    elif trainargs.optim == 'multiplicative':
        for name, wtype in zip(['pos', 'neg'], ['g+', 'g-']):
            weights = trainstate.optstate[wtype]
            act = jnp.dot(weights['linear']['w'].T, datapoint.reshape(-1, 1))
            idx = jnp.argsort(-act.ravel())

            filters = weights['linear']['w'].reshape(1, 28, 28, -1)
            filters = jnp.transpose(filters, [3, 0, 1, 2])

            if name == 'pos':            
                activated_filters = filters[idx[0:15]]
                activated_filters = jnp.concatenate([datapoint.reshape(1,1,28,28), activated_filters], axis=0)
            else:
                activated_filters = filters[idx[0:16]]
                
            grid = make_grid(torch.from_numpy(np.array(activated_filters)), 
                             nrow=8, normalize=True, scale_each=True, pad_value=1.0, padding=1)
            save_image(grid, os.path.join(args.resultsfolder, f'activated_filters_{name}.png'))

if __name__ == '__main__':
    main() 
