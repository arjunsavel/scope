import jax
import jax.numpy as jnp


@jax.jit
def calc_ccf(model_flux, data_arr_slice, n_pixel):
    """
    Calculates the CCF between a model and a data slice.

    Inputs
    ------
        :model_flux: (jnp.ndarray) The model flux, normalized and such.
        :data_arr_slice: (jnp.ndarray) The data slice, normalized and such.
        :n_pixels: (int) The number of pixels in the data slice.

    Outputs
    -------
        :logl: (float) The log-likelihood of the data given the model.
        :CCF: (float) The CCF between the model and the data.
    """
    model_vector = jnp.subtract(
        model_flux, jnp.vstack(jnp.mean(model_flux, axis=1))
    )  # normalized and such
    variance_model = jnp.var(model_vector, axis=1)
    variance_data = jnp.var(data_arr_slice, axis=1)
    cross_variance = (model_vector * data_arr_slice).sum(axis=1)

    # now need to sum
    CCF = (cross_variance / jnp.sqrt(variance_data * variance_model)).sum()

    logl = (
        -0.5 * n_pixel * jnp.log(variance_data + variance_model - 2.0 * cross_variance)
    ).sum()

    return logl, CCF


calc_ccf_map = jax.vmap(calc_ccf, in_axes=(0, 0, None), out_axes=0)
