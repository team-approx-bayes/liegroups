import argparse
import os
import sys
import pickle

import numpy as np
import torch
from tqdm import trange, tqdm

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp
import haiku as hk

from data import dataloader
from models import get_model
from util import tprint, nll_categorical, regularize_squared_l2
from optim import additive_optimizer, multiplicative_optimizer, affine_optimizer
from noise import gaussiannoise, rayleighnoise, uniformnoise, laplacenoise

num_workers = 4

# q0 distributions to be pushed around
noisegenerators = {
    'rayleigh' : rayleighnoise,
    'uniform' : uniformnoise,
    'gaussian' : gaussiannoise,
    'laplace' : laplacenoise
}

def get_optimizer(args, ndata, modelinit, modelapply, params, datapoint, initkey):
    wdecay = args.priorprec / (float(ndata) * args.dafactor) # weight-decay
    def regularized_nll(param, minibatch):
        logits = modelapply(param, None, minibatch[0])
        loss = nll_categorical(logits, minibatch[1])
        loss += wdecay * regularize_squared_l2(param)

        return loss

    if args.optim == 'sgd':
        optinit, optstep, optsample, optweights = additive_optimizer(
            jax.value_and_grad(regularized_nll, has_aux=False),
            learningrate = args.alpha1, 
            momentum = args.beta1,
            noisegenerator = lambda x, _: x,  # sgd is just additive rule w/o noise
            batchsplit = args.batchsplit, 
            mcsamples = args.mc)
        
    elif args.optim == 'additive':
        optinit, optstep, optsample, optweights = additive_optimizer(
            jax.value_and_grad(regularized_nll, has_aux=False),
            learningrate = args.alpha1, 
            momentum = args.beta1,
            noisegenerator = lambda x, k: noisegenerators[args.noise](x, k, args.noiseconfig),
            batchsplit = args.batchsplit, 
            mcsamples = args.mc)
        
    elif args.optim == 'multiplicative': 
        optinit, optstep, optsample, optweights = multiplicative_optimizer(
            jax.value_and_grad(regularized_nll, has_aux=False),
            learningrate = args.alpha1,
            momentum = args.beta1,
            noisegenerator = lambda x, k: noisegenerators[args.noise](x, k, args.noiseconfig, args.mc),
            temperature = args.temperature / (float(ndata) * args.dafactor), 
            mcsamples = args.mc,
            batchsplit = args.batchsplit,
            init_offset = args.multinitoffset)
        
        params = (params, modelinit(initkey, datapoint))
    
    elif args.optim == 'affine':
        optinit, optstep, optsample, optweights = affine_optimizer(
            lossgrad = jax.value_and_grad(regularized_nll, has_aux=False),
            alpha1 = args.alpha1,
            alpha2 = args.alpha2,
            beta1 = args.beta1,
            beta2 = args.beta2,
            noisegenerator = lambda x, k: noisegenerators[args.noise](x, k, args.noiseconfig),
            temperature = args.temperature / (float(ndata) * args.dafactor), 
            mcsamples = args.mc,
            batchsplit = args.batchsplit,
            initA = args.custominit)

    else: 
        print(f'Optimizer {args.optim} not implemented.')
        sys.exit()

    return optinit, optstep, optsample, optweights, params

def main():
    """ training loop """

    # options and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomseed', dest='randomseed', type=int, default=0)
    parser.add_argument('--alpha1', dest='alpha1', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--alpha2', dest='alpha2', type=float, default=0.01,
                        help='separate learning rate for A (only affine rule)')
    parser.add_argument('--optim', dest='optim', type=str, default='sgd')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, 
                        help='momentum for weight-gradient')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999,
                        help='momentum for variance-gradient')
    parser.add_argument('--priorprec', dest='priorprec', type=float, 
                        default=25.0, help='prior precision')
    parser.add_argument('--temperature', dest='temperature', type=float, 
                        default=1.0, help='scaling of entropy-term')
    parser.add_argument('--dafactor', dest='dafactor', type=float, default=1.0,
                        help='multiplicative factor to adjust size of dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=200)
    parser.add_argument('--testbatchsize', dest='testbatchsize', 
                        type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=180)
    parser.add_argument('--warmup', dest='warmup', type=int, default=5,
                        help='linear learning-rate warmup')
    parser.add_argument('--dataset', dest='dataset', 
                        type=str, default='mnist')
    parser.add_argument('--mc', dest='mc', type=int, default=1, 
                        help='number of mcsamples during training')
    parser.add_argument('--batchsplit', dest='batchsplit', type=int, default=1,
                        help='independent perturbations on subbatches?')
    parser.add_argument('--noaugment', dest='augment', action='store_false',
                        help='no data augmentation')
    parser.add_argument('--model', dest='model', default='resnet20_frn',
                        help='model architecture')
    parser.add_argument('--datasetfolder', dest='datasetfolder', type=str, 
                        default='datasets')
    parser.add_argument('--resultsfolder', dest='resultsfolder', type=str, 
                        default='results')
    parser.add_argument('--noise', dest='noise', type=str, 
                        default='gaussian', help='noise distribution')
    parser.add_argument('--noiseconfig', dest='noiseconfig', type=float, 
                        default=1.0, help='hyperparam of noise distribution')
    parser.add_argument('--custominit', dest='custominit', type=float, 
                        default=1.0, help='special initialization value for variance')
    parser.add_argument('--multinitoffset', dest='multinitoffset', type=float, 
                        default=1e-5, help='initialization offset')
    
    parser.set_defaults(augment = True)
    args = parser.parse_args()

    idx = 0 
    while True: 
        outpath = f"""{args.resultsfolder}/{args.dataset}_{args.model}/{args.optim}/run_{idx}"""
        if not os.path.exists(outpath):
            break 

        idx += 1 

    os.makedirs(outpath)

    print('information of this training run')
    print('\n'.join(f'  > {k}={v}' for k, v in args.__dict__.items()))
    print(f'  > results are saved in {outpath}.')

    # fix randomseeds
    rngkey = jax.random.PRNGKey(args.randomseed)
    #np.random.seed(args.randomseed)
    torch.manual_seed(args.randomseed) 

    # prepare dataset 
    try:
        trainset, testset, trainloader, testloader = \
            dataloader(args.dataset)(args.batchsize, args.testbatchsize, 
                                     args.datasetfolder, args.augment, num_workers)
        
    except KeyError:
        print(f'Dataset {args.dataset} not implemented.')
        sys.exit()

    ndata = len(trainset)
    ntestdata = len(testset)
    nclasses = len(trainset.classes)

    print(f"""  > dataset={args.dataset} (ntrain={ndata}, """
        f"""ntest={ntestdata}, nclasses={nclasses})""")

    # prepare model
    modelapply, modelinit = get_model(args.model.lower(), nclasses)
    modelapply = jax.jit(modelapply)
    rngkey, initkey = jax.random.split(rngkey)

    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)
    print('  > datashape (minibatch) ', datapoint.shape)
    

    params = modelinit(initkey, datapoint)
    numparams = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"""  > model='{args.model}' ({numparams} parameters)""")

    # prepare optimizer
    rngkey, initkey = jax.random.split(rngkey)
    optinit, optstep, optsample, optweights, params = get_optimizer(
        args, ndata, modelinit, modelapply, params, datapoint, initkey 
    )

    def train_epoch(trainstate, lrfactor): 
        losses = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            X = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            y = jax.nn.one_hot(targets.numpy(), nclasses)

            trainstate, loss = optstep(trainstate, (X, y), lrfactor)
            losses.append(float(loss))

        return trainstate, jnp.mean(jnp.array(losses))

    def testestimate(trainstate): 
        correct = 0
        total = 0
        test_nll = 0

        for batch_idx, (inputs, targets) in enumerate(testloader):
            dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            tgt = jax.nn.one_hot(targets.numpy(), nclasses)

            theta = optweights(trainstate)
            logitsmean = modelapply(theta, None, dat)

            correct += jnp.sum(logitsmean.argmax(axis=1) == tgt.argmax(axis=1))
            test_nll += nll_categorical(logitsmean, tgt)
            total += logitsmean.shape[0]    

        return float(correct) / float(total), test_nll / float(total)

    trainstate = optinit(params, rngkey)
    optstep = jax.jit(optstep)

    # main loop
    total_time = 0.0 
    with open(os.path.join(outpath, 'info.txt'), 'wt', encoding='utf-8') as file:
        file.write('\n'.join(f'{k}={v}' for k,
                    v in args.__dict__.items()))
        file.write('\n')

    for epoch in trange(args.epochs + 1, 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1.):
        # learning rate scheduler
        if epoch < args.warmup:
            lrfactor = jnp.linspace(0.0, 1.0, args.warmup + 1)[epoch + 1]
        else:
            step_t = float(epoch - args.warmup) / float(args.epochs + 1 - args.warmup)
            lrfactor = 0.5 * (1.0 + jnp.cos(jnp.pi * step_t))

        # train one epoch
        trainstate, loss = train_epoch(trainstate, lrfactor)

        # save intermediate results
        acc, test_nll = testestimate(trainstate) 
        acc = acc * 100.0
        tprint(f"""[{epoch:3d}/{args.epochs}] Trainloss (at samples): {loss:.3f}"""
               f""" | TestNLL: {test_nll:.3f} Acc: {acc:.3f} """)
            
        with open(os.path.join(outpath, 'trainstate.pickle'), 'wb') as file:
            pickle.dump(trainstate, file)
            pickle.dump(args, file)

        with open(os.path.join(outpath, 'info.txt'), 'a', encoding='utf-8') as file:
            file.write(f"""[{epoch:3d}/{args.epochs}] Trainloss (at samples): {loss:.3f}"""
                       f""" | TestNLL: {test_nll:.3f} Acc: {acc:.3f} """)

if __name__ == '__main__':
    main()
