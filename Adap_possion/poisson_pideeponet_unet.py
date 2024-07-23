# In[]
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from deepxde import utils
from deepxde.backend import tf
from tensorflow import keras
from tensorflow.keras import layers

# dde.config.disable_xla_jit()
#dde.config.set_default_float("float32")

filebase = "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/PIunet"


# In[]
# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)  # b*h*w*unit
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        # equivalent: [hw X C] * [hw X C]^T, eliminate the index of c
        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
        # equivalent: [hw X hw] * [hw X c]
        proj = tf.einsum(
            "bhwHW,bHWc->bhwc", attn_score, v
        )  # how to use attention layer for imags
        proj = self.proj(proj)
        return inputs + proj


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        # temb = activation_fn(t)
        # temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
        #     :, None, None, :
        # ]  # bX1X1Xwidth, width=c

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        # x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        # x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2DTranspose(
            width, kernel_size=3, strides=(2, 2), padding="same"
        )(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def build_model(
    img_size=(128, 128),
    output_size=None,
    img_channels=1,
    widths=[8, 16, 32, 64],
    has_attention=[False, False, True, True],
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    if output_size is None:
        output_size = img_size[0] * img_size[1] * img_channels
    image_input = layers.Input(
        shape=(img_size[0] * img_size[1] * img_channels,), name="image_input"
    )
    x = layers.Reshape((img_size[0], img_size[1], img_channels))(image_input)
    x = layers.Conv2D(
        widths[0],
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)

    # temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    # temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )(x)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(x)
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(x)

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )(x)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 3), padding="valid", kernel_initializer=kernel_init(0.0))(
        x
    )
    x = layers.Flatten()(x)
    x = layers.Dense(output_size, kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input], x, name="unet")


# In[]


def equation(x, y, v):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0, component=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1, component=0)
    ##return -dy_xx - dy_yy - v
    return 0.01 * (dy_xx + dy_yy) + v


def boundary(_, on_boundary):
    return on_boundary


##geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

#### num_domain are # of points sampled inside domain, num_boundary # of points sampled on BC,
#### num_test=None all points inside domain are used for testing PDE residual loss
pde = dde.data.PDE(
    geom, equation, bc, num_domain=10000, num_boundary=396, num_test=None
)


# Function space
##func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
func_space = dde.data.GRF2D(length_scale=0.1, interp="linear", N=100)

### random source funcion is evaluated on 64 x 64 grid, i.e m = 50x50=2500
x = np.linspace(0, 1, num=64)
y = np.linspace(0, 1, num=64)
xv, yv = np.meshgrid(x, y)
eval_pts = np.vstack((np.ravel(xv), np.ravel(yv))).T

### 5000 random distributions, or 5000 traning samples, 100 testing samples
data = dde.data.PDEOperatorCartesianProd(
    pde,
    func_space,
    eval_pts,
    5000,
    function_variables=None,
    num_test=100,
    batch_size=32,
)
# In[]
# Net branch first, trunk second
branch_model = build_model(
    img_size=(64, 64),
    output_size=256,
    img_channels=1,
    widths=[8, 16, 32, 64],
    has_attention=[False, False, True, True],
    num_res_blocks=2,
    norm_groups=8,
)

# net = dde.nn.DeepONetCartesianProd(
#     [2500, 256, 256, 256, 256],
#     [2, 256, 256, 256, 256],
#     "tanh",
#     "Glorot normal",
# )

net = dde.nn.DeepONetCartesianProd(
    [None,branch_model,256],
    [2, 256, 256, 256, 256],
    "tanh",
    "Glorot normal",
)
net.num_trainable_parameters()

# In[]
"""
def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)


net.apply_feature_transform(periodic)
"""

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
check_point_filename = os.path.join(filebase, "model.ckpt")
# checkpointer = dde.callbacks.ModelCheckpoint(check_point_filename, verbose=1, save_better_only=True)
losshistory, train_state = model.train(
    iterations=60000, model_save_path=check_point_filename
)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# In[]
import time as TT

st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print("y_pred.shape =", y_pred.shape)

print("Inference took ", duration, " s")
print("Prediction speed = ", duration / float(len(y_pred)), " s/case")


##xy_test = geom.uniform_points(10000, boundary=True)

x_s = np.linspace(0, 1, 64)
y_s = np.linspace(0, 1, 64)
XX_s, YY_s = np.meshgrid(x_s, y_s)
xy_test = np.vstack((np.ravel(XX_s), np.ravel(YY_s))).T

n = 3
features = func_space.random(n)
fx_test = func_space.eval_batch(features, eval_pts)
y_test = model.predict((fx_test, xy_test))


# In[]


def LaplaceOperator(x, y):
    dydx2 = dde.grad.hessian(y, x, i=0, j=0)
    dydy2 = dde.grad.hessian(y, x, i=1, j=1)
    return dydx2 + dydy2


class EvaluateDerivatives:
    """Generates the derivative of the outputs with respect to the trunck inputs.

    Args:
        model: DeepOnet.
        operator: Operator to apply to the outputs for derivative.
    """

    def __init__(self, model, operator):
        self.op = operator
        self.model = model

        @tf.function
        def op(inputs):
            y = self.model.net(inputs)
            # QB: inputs[1] is the input of the trunck
            # QB: y[0] is the output corresponding
            # to the first input sample of the branch input,
            # each time we only consider one sample
            return self.op(inputs[1], y[0][:, None])

        self.tf_op = op

    def eval(self, inputs):
        self.value = []
        input_branch, input_trunck = inputs
        for inp in input_branch:
            x = (inp[None, :], input_trunck)
            self.value.append(utils.to_numpy(self.tf_op(x)))
        self.value = np.array(self.value)

    def get_values(self):
        return self.value
