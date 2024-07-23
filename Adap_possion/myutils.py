# In[]
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import deepxde as dde
from deepxde import utils
from deepxde.backend import tf


# %%
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
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        # x = layers.Conv2DTranspose(
        #     width, kernel_size=3, strides=(2, 2), padding="same"
        # )(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply

def UNET(
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
    #x = AttentionBlock(widths[-1], groups=norm_groups)(x)
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
    x=layers.Conv2D(1, (3, 3),strides=2, padding="valid", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Flatten()(x)
    x=layers.Dense(10, kernel_initializer=kernel_init(0.0))(x)
    x = layers.Dense(output_size, kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input], x, name="unet")


# %%
def find_checkpoint_2restore(filebase):
    # Initialize an empty dictionary to store the data
    filename = os.path.join(filebase, "checkpoint")
    if os.path.exists(filename):
        data = {}

        # Open the file
        with open(filename, "r") as file:
            # Iterate over each line in the file
            for line in file:
                # Strip any leading/trailing whitespace and split the line at ':'
                key, value = line.strip().split(":")
                # Add the key-value pair to the dictionary
                key = key.strip()
                value = value.strip().strip('"')
                data[key] = value
        restore_filename = data["model_checkpoint_path"]
        return os.path.join(filebase, restore_filename)
    else:
        return None


# %%
def LaplaceOperator2D(x, y):
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
            y = self.model(inputs)
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
        return self.value

    def get_values(self):
        return self.value

# %%
