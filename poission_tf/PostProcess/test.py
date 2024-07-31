# %%
def fun(layer_sizes_branch, layer_sizes_trunk, activation="tanh",apply_activation_outlayer=True):
    print('layer_sizes_branch',layer_sizes_branch)
    print('layer_sizes_trunk',layer_sizes_trunk)
    print('activation',activation)
    print('apply_activation_outlayer',apply_activation_outlayer)

def fun2(*args, **kwargs):
    fun(*args, **kwargs)
    
# %%
fun2([100*100, 100, 100, 100, 100, 100, 100,],
    [2, 100, 100, 100, 100, 100, 100,100],
    {"branch": "tf.keras.activations.swish", "trunk": "tanh"},apply_activation_outlayer=False)

# %%
