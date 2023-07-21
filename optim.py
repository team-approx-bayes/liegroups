""" lie-group optimizers given in appendix of the paper """

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from copy import deepcopy

class TrainState(NamedTuple):
    """
    collects the all the state required for neural network training
    """
    optstate: dict
    rngkey: None

def additive_optimizer(lossgrad,
                       learningrate : float, 
                       momentum : float,
                       noisegenerator,
                       mcsamples : int = 1,
                       batchsplit : int = 1): 
    """ additive group update with momentum (see Algorithm 1 in Appendix A.3)"""

    def init(weightinit, rngkey):
        optstate = dict()
        optstate['w'] = deepcopy(weightinit)
        optstate['gm'] = jax.tree_map(lambda p : jnp.zeros(shape = p.shape), weightinit)
        optstate['alpha'] = learningrate

        return TrainState(
            optstate = optstate,
            rngkey = rngkey)

    def _expectation(k, w, X_split, y_split, allrng, carry):
        """ approximate expectation by random sampling """

        xi = noisegenerator(w, allrng[k])
        randomparam = jax.tree_map(lambda p, x: p + x, w, xi)
        sampleloss, samplegrad = \
            lossgrad(randomparam, (X_split[k % batchsplit], y_split[k % batchsplit]))

        avggrad = jax.tree_map(lambda x, u: (k * x + u) / (k + 1.0), \
            carry[0], samplegrad)
        avgloss = (k * carry[1] + sampleloss) / (k + 1.0)     

        return avggrad, avgloss

    def step(trainstate, minibatch, lrfactor):
        """ perform one update step """

        optstate = trainstate.optstate

        X_split = minibatch[0].reshape(batchsplit, -1, *minibatch[0].shape[1:])
        y_split = minibatch[1].reshape(batchsplit, -1, *minibatch[1].shape[1:]) 

        allrng = jax.random.split(trainstate.rngkey, batchsplit * mcsamples + 1)

        xi = noisegenerator(optstate['w'], allrng[0])
        randomparam = jax.tree_map(lambda p, x: p + x, optstate['w'], xi)
        avgloss, avggrad = \
            lossgrad(randomparam, (X_split[0], y_split[0]))

        # potentially use more samples
        avggrad, avgloss = \
            jax.lax.fori_loop(1, batchsplit * mcsamples, \
                lambda k, carry : _expectation(k, optstate['w'], X_split, y_split, allrng, carry), \
                (avggrad, avgloss))      

        optstate['gm'] = jax.tree_map(
            lambda gm, g: momentum * gm + (1 - momentum) * g,
            optstate['gm'],
            avggrad)

        optstate['w'] = jax.tree_map(
            lambda gm, w: w - lrfactor * optstate['alpha'] * gm,
            optstate['gm'],
            optstate['w'])

        newtrainstate = trainstate._replace(
            optstate = optstate,
            rngkey = allrng[-1])

        return newtrainstate, avgloss

    def sample(trainstate): 
        """ generate a random sample from the estimated distribution """

        allrng = jax.random.split(trainstate.rngkey, 2)
        xi = noisegenerator(trainstate.optstate['w'], allrng[0])
        randomparam = jax.tree_map(lambda p, x: p + x, trainstate.optstate['w'], xi)

        newtrainstate = trainstate._replace(
            rngkey = allrng[-1])

        return newtrainstate, randomparam
    
    def detweights(trainstate):
        """ return some deterministic weight estimate """

        return trainstate.optstate['w']

    return init, step, sample, detweights

def multiplicative_optimizer(lossgrad,
                             learningrate : float, 
                             momentum : float,
                             noisegenerator,
                             temperature : float = 1,
                             mcsamples : int = 1,
                             batchsplit : int = 1, 
                             init_offset : float = 1e-5): 
    """ multiplicative group update with momentum (see Algorithm 2 in Appendix A.4) """

    def init(weightinit, rngkey):
        optstate = dict()

        # we keep two sets of weights, corresponding to excitatory (g+)
        # and inhibitory neurons (g-). 
        optstate['g+'] = jax.tree_map(lambda p : jnp.abs(p) + init_offset, weightinit[0])
        optstate['g-'] = jax.tree_map(lambda p : jnp.abs(p) + init_offset, weightinit[1])
        optstate['gm+'] = jax.tree_map(lambda p : jnp.zeros(shape = p.shape), weightinit[0])
        optstate['gm-'] = jax.tree_map(lambda p : jnp.zeros(shape = p.shape), weightinit[1])
        optstate['alpha'] = learningrate

        return TrainState(
            optstate = optstate,
            rngkey = rngkey)

    def _expectation(k, wp, wm, X_split, y_split, noiseplus, noiseminus, carry):
        """ approximate expectation by random sampling """

        perturbed_plus = jax.tree_map(lambda gp, np : gp * np[k], wp, noiseplus)
        perturbed_minus = jax.tree_map(lambda gm, nm : gm * nm[k], wm, noiseminus)

        sampleloss, samplegrad = \
            lossgrad(jax.tree_map(lambda pp, pm: pp - pm, perturbed_plus, perturbed_minus),
                (X_split[k % batchsplit], y_split[k % batchsplit]))

        avggrad_plus = jax.tree_map(lambda x, u, pp: 
                                    (k * x + (u * pp)) / (k + 1.0), \
                                    carry[0], samplegrad, perturbed_plus)

        avggrad_minus = jax.tree_map(lambda x, u, pm: 
                                    (k * x + (-u * pm)) / (k + 1.0), \
                                    carry[1], samplegrad, perturbed_minus)
        
        avgloss = (k * carry[2] + sampleloss) / (k + 1.0)     

        return avggrad_plus, avggrad_minus, avgloss

    def step(trainstate, minibatch, lrfactor):
        """ perform one update step """

        optstate = trainstate.optstate

        X_split = minibatch[0].reshape(batchsplit, -1, *minibatch[0].shape[1:])
        y_split = minibatch[1].reshape(batchsplit, -1, *minibatch[1].shape[1:]) 

        noiseplus = noisegenerator(optstate['g+'], trainstate.rngkey) 
        noiseminus = noisegenerator(optstate['g-'], trainstate.rngkey) 

        dummycarry = (optstate['gm+'], optstate['gm-'], 0.0)
        avggrad_plus, avggrad_minus, avgloss = _expectation(
            0, optstate['g+'], optstate['g-'], 
            X_split, y_split, noiseplus, noiseminus, dummycarry) 
        
        # potentially use more samples
        avggrad_plus2, avggrad_minus2, avgloss2 = \
            jax.lax.fori_loop(1, batchsplit * mcsamples, \
                lambda k, carry : _expectation(k, optstate['g+'], optstate['g-'],
                                               X_split, y_split, noiseplus, noiseminus, carry), \
                (avggrad_plus, avggrad_minus, avgloss))      

        # exponential smoothing of stochastic gradients
        optstate['gm+'] = jax.tree_map(
            lambda gm, grad: momentum * gm + (1.0 - momentum) * (grad - temperature),
            optstate['gm+'],
            avggrad_plus2)

        optstate['gm-'] = jax.tree_map(
            lambda gm, grad: momentum * gm + (1.0 - momentum) * (grad - temperature),
            optstate['gm-'],
            avggrad_minus2)
        
        # lie-group update
        optstate['g+'] = jax.tree_map(
            lambda gm, g: g * jnp.exp(-optstate['alpha'] * lrfactor * gm),
            optstate['gm+'],
            optstate['g+'])

        optstate['g-'] = jax.tree_map(
            lambda gm, g: g * jnp.exp(-optstate['alpha'] * lrfactor * gm),
            optstate['gm-'],
            optstate['g-'])

        newtrainstate = trainstate._replace(
            optstate = optstate)

        return newtrainstate, avgloss2

    def sample(trainstate): 
        """ generate a random sample from the estimated distribution """
        
        noiseplus = noisegenerator(trainstate.optstate['g+'], trainstate.rngkey) 
        noiseminus = noisegenerator(trainstate.optstate['g-'], trainstate.rngkey) 

        randomparam = jax.tree_map(lambda gp, gm, np, nm: gp * np - gm * nm, 
                                   trainstate.optstate['g+'], trainstate.optstate['g-'],
                                   noiseplus, noiseminus)

        newtrainstate = trainstate

        return newtrainstate, randomparam

    def detweights(trainstate):
        """ return some deterministic weight estimate """

        return jax.tree_map(lambda gp, gm: gp - gm, 
                            trainstate.optstate['g+'], trainstate.optstate['g-'])

    return init, step, sample, detweights

def affine_optimizer(
        lossgrad,
        alpha1 : float,
        alpha2 : float,
        beta1 : float,
        beta2 : float,
        noisegenerator,
        temperature : float = 1.0,
        mcsamples : int = 1,
        batchsplit : int = 1,
        initA : float = 1.0):
    """ 
    affine group update with momentum (see Algorithm 3 in Appendix A.5)
    """

    def init(weightinit, rngkey):
        optstate = dict()

        optstate['A'] = jax.tree_map(lambda p : initA * jnp.ones(shape = p.shape), weightinit)
        optstate['b'] = jax.tree_map(lambda p : p, weightinit) 
        optstate['MU'] = jax.tree_map(lambda p : jnp.zeros(shape = p.shape), weightinit)
        optstate['MV'] = jax.tree_map(lambda p : jnp.zeros(shape = p.shape), weightinit)
        optstate['alpha1'] = alpha1
        optstate['alpha2'] = alpha2

        return TrainState(
            optstate = optstate,
            rngkey = rngkey)

    def _expectation(k, A, b, X_split, y_split, allrng, carry):
        """ approximate expectation by random sampling """

        epsi = noisegenerator(b, allrng[k])
        randomparam = jax.tree_map(lambda b, A, e: b + A * e, b, A, epsi)
        sampleloss, samplegrad = \
            lossgrad(randomparam, (X_split[k % batchsplit], y_split[k % batchsplit]))

        avggrad_U = jax.tree_map(lambda x, u, A, e: (k * x + A * e * u) / (k + 1.0), \
            carry[0], samplegrad, A, epsi)
        avggrad_V = jax.tree_map(lambda x, u, A: (k * x + A * u) / (k + 1.0), \
            carry[1], samplegrad, A)
        avgloss = (k * carry[2] + sampleloss) / (k + 1.0)     

        return avggrad_U, avggrad_V, avgloss

    def step(trainstate, minibatch, lrfactor):
        """ perform one update step """

        optstate = trainstate.optstate

        X_split = minibatch[0].reshape(batchsplit, -1, *minibatch[0].shape[1:])
        y_split = minibatch[1].reshape(batchsplit, -1, *minibatch[1].shape[1:]) 

        allrng = jax.random.split(trainstate.rngkey, batchsplit * mcsamples + 1)

        dummycarry = (optstate['MU'], optstate['MV'], 0.0)
        avggrad_U, avggrad_V, avgloss = _expectation(
            0, optstate['A'], optstate['b'], 
            X_split, y_split, allrng, dummycarry) 
        
        # potentially use more samples
        avggrad_U, avggrad_V, avgloss = \
            jax.lax.fori_loop(1, batchsplit * mcsamples, \
                lambda k, carry : _expectation(k, optstate['A'], optstate['b'],
                                               X_split, y_split, allrng, carry), \
                (avggrad_U, avggrad_V, avgloss))      

        optstate['MU'] = jax.tree_map(
            lambda mu, gu: beta2 * mu + (1 - beta2) * (gu - temperature),
            optstate['MU'],
            avggrad_U)

        optstate['MV'] = jax.tree_map(
            lambda mv, v: beta1 * mv + (1 - beta1) * v,
            optstate['MV'],
            avggrad_V)

        # group-exponential
        def grpexp(rho, x):
            return jax.lax.select(
                jnp.abs(x) < 1e-6, -rho + x * (rho ** 2.0) / 2, 
                (jnp.exp(-rho * x) - 1) / x)

        optstate['b'] = jax.tree_map(
            lambda a, b, mu, mv : b + a * mv * grpexp(lrfactor * optstate['alpha1'], mu),
            optstate['A'],
            optstate['b'], 
            optstate['MU'],
            optstate['MV'])

        optstate['A'] = jax.tree_map(
            lambda a, mu : a * jnp.exp(-lrfactor * optstate['alpha2'] * mu),
            optstate['A'], 
            optstate['MU'])  

        newtrainstate = trainstate._replace(
            optstate = optstate,
            rngkey = allrng[-1])

        return newtrainstate, avgloss

    def sample(trainstate): 
        """ generate a random sample from the estimated distribution """
        allrng = jax.random.split(trainstate.rngkey, 2)
        optstate = trainstate.optstate
        epsi = noisegenerator(optstate['b'], allrng[0])
        randomparam = jax.tree_map(lambda b, A, e: b + A * e, 
                                   optstate['b'], optstate['A'], epsi)

        newtrainstate = trainstate._replace(
            rngkey = allrng[-1])

        return newtrainstate, randomparam

    def detweights(trainstate):
        """ return some deterministic weight estimate """

        return trainstate.optstate['b']

    return init, step, sample, detweights
