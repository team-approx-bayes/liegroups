from tqdm import tqdm
import jax 
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def tprint(obj):
    """ helper to print training progress """
    tqdm.write(str(obj))

def nll_categorical(logits, labels):
    """ multiclass classification negative log-likelihood """

    loss = -jnp.sum(logits * labels, axis = 1) + logsumexp(logits, axis = 1)
    return jnp.mean(loss, axis = 0)

def regularize_squared_l2(params):
    """ squared l2-norm regularization """

    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

def ece(probs, y_batch, bins=20):
    """ expected calibration error, 
        source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py """

    bin_boundaries = jnp.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = jnp.max(probs, 1)
    predictions = jnp.argmax(probs, 1)
    accuracies = (predictions == jnp.argmax(y_batch, 1))

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) * (confidences <= bin_upper)
        prob_in_bin = in_bin.astype('float32').mean()

        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].astype('float32').mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece 
