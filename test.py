import argparse
import os
import pickle
import sys

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from data import dataloader
from models import get_model
from train import get_optimizer
from util import ece

num_workers = 4

def main():
    # options and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testbatchsize', dest='testbatchsize', 
                        type=int, default=200)
    parser.add_argument('--testmc', dest='testmc', type=int, default=32,
                        help='number of mcsamples used in bayesian model averaging')
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
                                          trainargs.datasetfolder, trainargs.augment, num_workers)
        
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

    # create model and optimizer
    modelapply, modelinit = get_model(trainargs.model.lower(), nclasses)
    modelapply = jax.jit(modelapply)

    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)
    print('  > datashape (minibatch) ', datapoint.shape)
    params = modelinit(initkey, datapoint)
    numparams = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"""  > model='{trainargs.model}' ({numparams} parameters)""")

    rngkey, initkey = jax.random.split(rngkey)
    _, __, optsample, optweights, params = get_optimizer(
        args=trainargs, ndata=ndata, modelinit=modelinit, 
        modelapply=modelapply, params=params, datapoint=datapoint, initkey=initkey)
    
    # evaluate model
    batchprobs_g = [] 
    batchprobs_bayes = [] 
    batchlabels = [] 

    nll_g = 0.0
    nll_bayes = 0.0 

    correct_g = 0
    correct_bayes = 0
    total = 0

    print('testing...')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
        tgt = jax.nn.one_hot(targets.numpy(), nclasses)

        theta = optweights(trainstate)
        logits_g = modelapply(theta, None, dat)
        correct_g += jnp.sum(logits_g.argmax(axis=1) == tgt.argmax(axis=1))
        total += logits_g.shape[0]    

        nll_g += -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logits_g, axis=1), axis=1))

        batchprobs_g.append(jax.nn.softmax(logits_g, axis=1))
        batchlabels.append(tgt)

        sampleprobs = [] 
        samplelogits = [] 
        for i in range(args.testmc):
            trainstate, theta_sampled = optsample(trainstate)
            logits = modelapply(theta_sampled, None, dat)
            samplelogits.append(logits) 
            sampleprobs.append(jax.nn.softmax(logits, axis=1))

        bayesprobs = jnp.mean(jnp.array(sampleprobs), axis=0)
        correct_bayes += jnp.sum(bayesprobs.argmax(axis=1) == tgt.argmax(axis=1))
        batchprobs_bayes.append(bayesprobs)

        temp = jax.nn.log_softmax(jnp.array(samplelogits), axis=2) 
        nll_bayes += jnp.mean(jnp.sum(-tgt * logsumexp(temp, b=1/args.testmc, axis=0), axis=1))

    testacc_g = 100.0 * (float(correct_g) / float(total))
    nll_g /= float(batch_idx) 
    ece_g = ece(jnp.concatenate(batchprobs_g, axis=0), 
                jnp.concatenate(batchlabels, axis=0))

    testacc_bayes = 100.0 * (float(correct_bayes) / float(total))
    nll_bayes /= float(batch_idx) 
    ece_bayes = ece(jnp.concatenate(batchprobs_bayes, axis=0), 
                    jnp.concatenate(batchlabels, axis=0))

    print('results at g:')
    print('  > testacc=%.2f%%, nll=%.4f, ece=%.4f' % (testacc_g, nll_g, ece_g))
    print('results at model average (%d samples):' % args.testmc)
    print('  > testacc=%.2f%%, nll=%.4f, ece=%.4f' % (testacc_bayes, nll_bayes, ece_bayes))

if __name__ == '__main__':
    main() 