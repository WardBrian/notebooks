import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # FWAM 2025: [JAX](https://docs.jax.dev/en/latest/) tutorial from DFM

    JAX as a stack of compilers: Python function -> Jaxpr -> transformations (vmap, AD, etc) -> XLA
    """
    )
    return


@app.cell
def _():
    import jax
    jax.config.update("jax_num_cpu_devices", 6)
    return (jax,)


@app.cell
def _():
    import numpy as np
    import jax.numpy as jnp
    # also jax.scipy, less complete
    return jnp, np


@app.cell
def _(jnp, np):
    x_np = np.linspace(0,1,5)
    x = jnp.linspace(0,1,5)
    return x, x_np


@app.cell
def _(jnp, x):
    jnp.sin(x)
    return


@app.cell
def _(x_np):
    x_np.dtype
    return


@app.cell
def _(jax):
    jax.config.update("jax_enable_x64", True) # global
    # or local using 
    # with jax.enable_x64():
    #     ...
    return


@app.cell
def _(jnp):
    jnp.linspace(0,1,5)
    return


@app.cell
def _(jax, jnp):
    @jax.jit
    def f(x):
        print("running f")
        y = jnp.sin(x)
        # jax.debug.print("{}", y)
        return 1.5 + jnp.exp(y)
    return (f,)


@app.cell
def _(f, x):
    f(x)
    return


@app.cell
def _(f, x):
    f(x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""jits are cached based on shape and dtype""")
    return


@app.cell
def _(f, x):
    f(x + 1)
    return


@app.cell
def _(f, x):
    f(x[:-1])
    return


@app.cell
def _(jax):
    counter = {"count": 1}

    @jax.jit
    def f2(x):
        counter['count'] += 1
        return x
    return counter, f2


@app.cell
def _(counter, f2, x):
    f2(x)
    print(counter)
    return


@app.cell
def _(jax, x):
    counter2 = {"count": 1}

    @jax.jit
    def f2_pure(counter, x):
        counter['count'] += 1
        return counter, x

    counter2, _ = f2_pure(counter2, x)
    counter2, _ = f2_pure(counter2, x)
    counter2, _ = f2_pure(counter2, x)
    print(counter2)
    return


@app.cell
def _(jax):
    @jax.jit
    def f3(x):
        print(x)
        if x < 0.5:
            return x + 1
        else:
            return x - 1
    # jax.lax.cond is what you would want to use
    # or, we can depend on np.shape(x) etc
    return (f3,)


@app.cell
def _(f3):
    f3(1.5)
    return


@app.cell
def _(f, jax):
    jax.make_jaxpr(f)(0.5)
    return


@app.cell
def _(f, jax, x):
    print(jax.jit(f).lower(x).as_text())
    return


@app.cell
def _(f, jax, x):
    print(jax.jit(f).lower(x).compile().as_text())
    return


@app.cell
def _(jax, jnp, x):
    def f_(x):
        y = jnp.sin(x) 
        return 1.5 + jnp.exp(y) - 0.5
    # note just the constant 1
    print(jax.jit(f_).lower(x).compile().as_text())
    return (f_,)


@app.cell
def _(mo):
    mo.md(r"""# `jax.grad`""")
    return


@app.cell
def _(f, jax):
    jax.grad(jax.jit(jax.grad(f)))(0.5)
    return


@app.cell
def _(f_, jax):
    print(jax.make_jaxpr(f_)(0.5))
    print(jax.make_jaxpr(jax.grad(f_))(0.5))
    return


@app.cell(hide_code=True)
def _(mo):
    n_items = mo.ui.slider(start=2, stop=100, step=2, label="number of points")
    n_items
    return (n_items,)


@app.cell
def _(f, jax, jnp, n_items):
    import matplotlib.pyplot as plt

    x_ = jnp.linspace(0,5,n_items.value)
    # vmap lets you do a vectorized map over elementwise functions
    val, grad = jax.vmap(jax.value_and_grad(f))(x_)
    grad2 = jax.vmap(jax.grad(jax.grad(f)))(x_)

    plt.plot(x_, val)
    plt.plot(x_, grad)
    plt.plot(x_, grad2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 102: sharding

    See [the docs](https://docs.jax.dev/en/latest/sharded-computation.html)
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
