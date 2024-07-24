# %%
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import Adaptive_DeepONet.Adap_possion.DeepONet_torch as DeepONet_torch 
import tensorflow as tf
# %%
degree = 3
space = dde.data.PowerSeries(N=degree + 1)
# Choose evaluation points
num_eval_points = 10
xy_coor = np.linspace(0, 1, num_eval_points)[:,None]

space_p1 = dde.data.PowerSeries(N=degree + 2)
num_function = 100
features = space_p1.random(num_function)
fx = space_p1.eval_batch(features, xy_coor)

# %%
class ICBCs():
    def __init__(self, xy_coor, fx):
        self.xy_coor = xy_coor
        self.fx = fx
        self.num_eval_points = xy_coor.shape[0]
        self.bcic_mask = np.isclose(xy_coor[:,0],np.max(xy_coor)) | np.isclose(xy_coor[:,0],np.min(xy_coor))

    
BCIC_mask = np.isclose(xy_coor[:,0],np.max(xy_coor)) | np.isclose(xy_coor[:,0],np.min(xy_coor))

# %%
@tf.function
def equation(x, y, f):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - f

class PDEsLosses():
    def __init__(self,eqs,model, BCIC_mask,name=None):
        #super().__init__(name=name)
        interior_mask = ~BCIC_mask
        self.interior_indices = np.where(interior_mask)[0]
        self.equations = eqs
        self.model = model
    def __call__(self, data,y_pred):
        input_branch,input_trunck,aux=data[0]
        #for (inp,v) in zip(input_branch,aux):
        
        x=(input_branch[:1,:],input_trunck)        
        y=self.model(x)
        loss=self.equations(input_branch[:1,:],y,aux[:1])
        
        return tf.reduce_mean(loss)  # MSE loss

class BCICLosses():
    def __init__(self,BCIC_mask,val,name=None):
        #super().__init__(name=name)
        self.BCIC_mask = BCIC_mask
        self.BCIS_indices = np.where(BCIC_mask)[0]
        self.val = val
    def __call__(self,data, y_pred):
        loss=tf.reduce_mean(tf.square(y_pred[self.BCIS_indices] - self.val))
        return loss
    

class PIDeepONetCartesianProd(DeepONet_torch.DeepONetCartesianProd):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="tanh",losses_fun=[]):
        super().__init__(layer_sizes_branch, layer_sizes_trunk, activation="tanh")
        self.losses_fun = losses_fun
        self.losses=[10.0**100]*len(losses_fun)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    def train_step(self,data):
        input_branch,input_trunck,aux=data[0]
        y_pred=self((input_branch,input_trunck))
        with tf.GradientTape() as tape:
            y_pred=self((input_branch,input_trunck))
            for i,ls_fun in enumerate(self.losses_fun):
                self.losses[i] = ls_fun(data,y_pred)
            total_loss = sum(self.losses)  # Aggregate the losses
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(self.losses[0])
        self.reconstruction_loss_tracker.update_state(self.losses[1])
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }



# %%
dim_x = 1
p = 32
model = DeepONet_torch.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [1, 32, p],
    activation="tanh"
)
pde_loss = PDEsLosses(equation,model, BCIC_mask,name='residual_loss')
bc_loss=BCICLosses(BCIC_mask,0.0,name='bc_loss')

optm = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optm, loss=[pde_loss,bc_loss])
# %%
x_train = (fx, xy_coor)
y_train = fx

data = DeepONet_torch.TripleCartesianProd(x_train, y_train, x_train, y_train, batch_size=64)

# %%
h = model.fit(
    data.train_dataset, epochs=20, verbose=2
)
# %%
