import jax 
import jax.numpy as jnp
import numpy as np

def gaussiannoise(param, key, std):
    treedef = jax.tree_util.tree_structure(param)
    num_vars = len(jax.tree_util.tree_leaves(param))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(lambda p, k: std * jax.random.normal(k, shape=p.shape), param,
                              jax.tree_util.tree_unflatten(treedef, all_keys[1:]))
    return noise

def rayleighnoise(param, key, _, mc):
    noise = jax.tree_map(
        lambda p: jnp.array(np.random.rayleigh(1.0, size=(mc, *p.shape))), 
        param)

    return noise

def uniformnoise(param, key, noisestrength):
    treedef = jax.tree_util.tree_structure(param)
    num_vars = len(jax.tree_util.tree_leaves(param))
    all_keys = jax.random.split(key, num=(num_vars + 1))

    noise = jax.tree_map(
        lambda p, k: noisestrength * (jax.random.uniform(key, shape=p.shape) - 0.5), 
        param,
        jax.tree_util.tree_unflatten(treedef, all_keys[1:]))

    return noise

def laplacenoise(param, key, noisestrength):
    treedef = jax.tree_util.tree_structure(param)
    num_vars = len(jax.tree_util.tree_leaves(param))
    all_keys = jax.random.split(key, num=(num_vars + 1))

    noise = jax.tree_map(
        lambda p, k: noisestrength * jax.random.laplace(k, shape=p.shape), 
        param,
        jax.tree_util.tree_unflatten(treedef, all_keys[1:]))

    return noise




