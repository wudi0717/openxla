import jax_plugins.musa as musa_plugin
musa_plugin.initialize()

import jax
import jax.numpy as jnp

print("devices:", jax.devices())

@jax.jit
def f(x):
    return x * 2 + 1

print("scalar:", jnp.add(1, 2))
print("jit:", f(jnp.array([1, 2, 3])))
