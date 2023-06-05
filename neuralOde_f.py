import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import torch as tr
from scipy.integrate import odeint
import torch.nn as nn

def make_data(x, target):
    data = []
    if len(x) != len(target):
        assert "Not equal length"
    for i in range(len(x)):
       data += [(x[i], target[i])] 
    return data

def make_points(x0, xn, h_x, y, shuffle = False, noise = False):
    x = np.array([x for x in np.arange(x0, xn + h_x, h_x)], dtype = np.float64)
    target = np.array([y(x) for x in x], dtype = np.float64)
    if noise: 
        noise = np.random.normal(0, 0.1, len(target))
        noise_target = target + noise
    else:
        noise_target = target
    # noise_target = target
    data = make_data(x, noise_target)
    if (shuffle): random.shuffle(data)
    return x, target, noise_target, data

def plot_points(obs=None, times=None, trajs=None, save=None, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], color = 'green', label  = 'true')
                # plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], label  = 'true')

    if trajs is not None: 
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], label = 'pred', lw=2.5, color = 'r')
        if save is not None:
            plt.savefig(save)
    plt.legend()
    plt.subplot(122)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.plot(o[:, b_i, 0], o[:, b_i, 1], color = 'green', label  = 'true')
                # plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], label = 'true')
    plt.legend()
    plt.show()

def to_np(x):
    return x.detach().clone().cpu().numpy()

def plot_points1(true_values=None, times=None, pred_values=None, x=None, figsize=(8, 4)):
    plt.figure(figsize=figsize)

    x_np = to_np(x)
    plt.subplot(121)
    if true_values is not None:
        y_np =  to_np(true_values)
        plt.scatter(x_np[:, 0], y_np[:, 0], color = 'green', label  = 'true')

    if pred_values is not None:
        y_np = to_np(pred_values)
        plt.scatter(x_np[:, 0], y_np[:, 0], color = 'red', label  = 'pred') 
    plt.legend()
    plt.subplot(122)
    if true_values is not None:
        y_np =  to_np(true_values)
        plt.plot(x_np[:, 0], y_np[:, 0], color = 'green', label  = 'true')

    if pred_values is not None:
        y_np = to_np(pred_values)
        plt.plot(x_np[:, 0], y_np[:, 0], color = 'red', label  = 'pred') 

    plt.legend()
    plt.show()

def make_batch(x_val, y_val, num):
    ind1 = 0
    ind2 = num
    l = []
    for i in range(len(x_val) // num):
        if ind2 >= len(x_val) - 1: break
        if ind1 + 1 == ind2:
            x = x_val[ind1:ind2].view(1, len(x_val[0]))
            y = y_val[ind1:ind2].view(1, len(y_val[0]))
            l += [(x, y)]
        else:
            l += [(x_val[ind1:ind2], y_val[ind1:ind2])]
            
        ind1 = ind2
        ind2 = ind1 + num
    if len(x_val) - (len(x_val) // num) * num != 0:
        l += [(x_val[len(x_val) - num:len(x_val)], y_val[len(x_val) - num:len(x_val)])]
    return l

def Plot_xy(data, labels = [], ax = None):
    if (ax != None):
        for i, xy in enumerate(data):
            ax.plot(xy[0], xy[1], '.', color = colors[i])
    else:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        for i, xy in enumerate(data):
            ax.plot(xy[0], xy[1], '.', color = colors[i])

def Plot2(X_true, Y_true, xy_noise, pred, t1 = ''):
    fig = plt.figure(figsize= (6, 6))
    plt.scatter(xy_noise[0], xy_noise[1], c = 'b', s = 5)
    plt.scatter(pred[0], pred[1], c = 'r', s = 4)
    plt.plot(X_true, Y_true, c = 'g', linewidth=0.7)
    plt.title(t1)
    plt.show()

def Plot3(X_true, Y_true, xy_noise, pred, t1 = ''):
    fig = plt.figure(figsize= (6, 6))
    plt.scatter(xy_noise[0], np.vstack(xy_noise[1]).T[0], c = 'b', s = 5)
    plt.scatter(xy_noise[0], np.vstack(xy_noise[1]).T[1], c = 'b', s = 5)
    plt.scatter(pred[0], np.vstack(pred[1]).T[0], c = 'r', s = 4)
    plt.scatter(pred[0], np.vstack(pred[1]).T[1], c = 'r', s = 4)
    plt.plot(X_true, Y_true.T[0], c = 'g', linewidth=0.7)
    plt.plot(X_true, Y_true.T[1], c = 'g', linewidth=0.7)
    plt.title(t1)
    plt.show()

class Solution:
    def __init__(self, h, lr, num_layer = -1):

        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.array([0.1, 0.1, 0.1])
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    
    def f(self, teta): return lambda t, z: teta[0] * z    
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * teta[0]
    def df_dteta(self, teta): return lambda t, z: np.array([z])

    def get_loss(self):
        return (self.target - self.z[-1])**2
    
    def der_loss_z(self, z, z_true):
        return -2 * (z_true - z)
    
    def test(self, z0_list):
        rez = []
        for z0 in z0_list:
            zn = self.forward(z0)
            rez += [[z0, zn]]
        return np.array(rez).T
    
    def forward(self, z0): 
        self.z0 = z0
        t_for_z = np.array(self.t)
        sol = odeint(self.f(self.teta), y0=self.z0, t=t_for_z, tfirst=True)
        self.z = sol[:, 0]
        # if len(self.z) != 21:
        #     print("len(self.z) = ", len(self.z))
        #     print("len(t_for_z) = ", len(t_for_z))
        return self.z[-1]

    def Get_tz(self, z0):
        sol = odeint(self.f(self.teta), y0=[z0], t=np.array(self.t), tfirst=True)     
        return [np.array(self.t), sol[:, 0]]

    def Make_dL(self):
        self.dL_dteta = np.zeros_like(self.teta)
        # print("self.dL_dteta = ", self.dL_dteta)
        f = lambda i: (self.a.T)[i] * self.df_dteta(self.teta)(self.h * i, self.z[i])
        for i in range(1, len(self.z) - 1):
            # print("f(i) = ", f(i))
            self.dL_dteta += f(i)

        self.dL_dteta += (f(0) + f(len(self.z) - 1)) / 2
        self.dL_dteta *= self.h
        # print("dfdTeta = ", self.dL_dteta)

    def backward(self, target):
        self.target = target
        loss = self.get_loss()
        self.losses.append(loss)
        
        a0 = self.der_loss_z(self.z[-1], self.target)

        t_for_a = np.array(self.t[::-1])
        # print("len(self.z) = ", len(self.z))
        # print("len(t_for_a) = ", len(t_for_a))
        # print("a0 = ", a0)
        sol = odeint(self.da_dt(self.teta), y0 = a0, t=t_for_a, tfirst=True, args=(self.z, t_for_a))
        # print("sol.a[-1] = ", sol[-1])
        self.a = sol[:, 0]

        self.Make_dL()

        pred_teta = self.teta
        self.lr = self.lr0

        self.teta = pred_teta - self.dL_dteta * self.lr
        t_for_z = np.array(self.t)
        sol = odeint(self.f(self.teta), y0 = self.z0, t=t_for_z, tfirst=True)
        self.z = sol[:, 0]
        new_loss = self.get_loss()

        return new_loss

    def Get_zz0t(self, list_z0):
        rez = []
        for i, z0 in enumerate(list_z0):
            r = []
            sol = odeint(self.f(self.teta), y0=[z0], t=np.array(self.t), tfirst=True)
            z = sol[:, 0]
            for i, t in enumerate(self.t):
                r += [[t, z[i]]]  
            rez += [np.array(r).T]
        return np.array(rez)

class Lin_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):
        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.random.normal(0.0, 0.0, 2)
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    
    def f(self, teta): return lambda t, z: teta[0] * z + teta[1]    
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * teta[0]
    def df_dteta(self, teta): return lambda t, z: np.array([z, 1])

class Sq2_ODENet(Solution):

    def __init__(self, h, lr, num_layer = -1):
        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.random.normal(0.0, 0.0, 3)
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) <= diff):
                diff = abs(t - val)
                pos = i
        # print("t = ", val, "pos = ", pos)
        return pos

    def f(self, teta): return lambda t, z: teta[2] * z**2 + teta[1] * z + teta[0]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * (teta[2] * 2 * z[self.Ind(t, list_t)] + teta[1])
    def df_dteta(self, teta): return lambda t, z: np.array([1, z, z**2])

class Sq3_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):
        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.random.normal(0.0, 0.0, 4)
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) <= diff):
                diff = abs(t - val)
                pos = i
        # print("t = ", val, "pos = ", pos)
        return pos

    def f(self, teta): return lambda t, z: teta[3] * z**3 + teta[2] * z**2 + teta[1] * z + teta[0]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * (teta[3] * 3 * z[self.Ind(t, list_t)]**2 + teta[2] * 2 * z[self.Ind(t, list_t)] + teta[1])
    def df_dteta(self, teta): return lambda t, z: np.array([1, z, z**2, z**3])

class Sq4_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):

        self.h = h
        self.t0, self.tn = 0.0, 1.0
        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        
        self.teta = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.losses = []
        self.lr = lr
        self.lr0 = lr
        self.num_layer = num_layer
    
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) < diff):
                diff = abs(t - val)
                pos = i
        return pos

    def f(self, teta): return lambda t, z:  teta[4] * z**4 + teta[3] * z**3 + teta[2] * z**2 + teta[1] * z + teta[0]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * (teta[4] * 4 * z[self.Ind(t, list_t)]**3 + teta[3] * 3 * z[self.Ind(t, list_t)]**2 + teta[2] * 2 *z[self.Ind(t, list_t)] + teta[1])
    def df_dteta(self, teta): return lambda t, z: np.array([1, z, z**2, z**3, z**4])
    
class Sq5_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):

        self.h = h
        self.t0, self.tn = 0.0, 1.0
        self.t = np.arange(self.t0, self.tn + self.h, self.h)

        self.teta = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.losses = []
        self.lr = lr
        self.lr0 = lr
        self.num_layer = num_layer
    
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) < diff):
                diff = abs(t - val)
                pos = i
        return pos

    def f(self, teta): return lambda t, z: teta[5] * z**5 + teta[4] * z**4 + teta[3] * z**3 + teta[2] * z**2 + teta[1] * z + teta[0]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * (teta[5] * 4 * z[self.Ind(t, list_t)]**4 + teta[4] * 4 * z[self.Ind(t, list_t)]**3 + teta[3] * 3 * z[self.Ind(t, list_t)]**2 + teta[2] * 2 *z[self.Ind(t, list_t)] + teta[1])
    def df_dteta(self, teta): return lambda t, z: np.array([1, z, z**2, z**3, z**4, z**5])

class Exp_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):
        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.random.normal(0.0, 0.0, 3)
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) <= diff):
                diff = abs(t - val)
                pos = i
        # print("t = ", val, "pos = ", pos)
        return pos

    def f(self, teta): return lambda t, z: teta[0] * math.exp(teta[1]*z) + teta[2]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * teta[0] * teta[1] * math.exp(teta[1]*z[self.Ind(t, list_t)])
    def df_dteta(self, teta): return lambda t, z: np.array([math.exp(teta[1]*z), teta[0] * z * math.exp(teta[1]*z), 1])

class Sin_ODENet(Solution):
    def __init__(self, h, lr, num_layer = -1):

        self.h = h
        self.t0, self.tn = 0.0, 1.0
        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = np.array([0.1, 0.1, 0.1])
        self.losses = []
        self.lr = lr
        self.lr0 = lr
        self.num_layer = num_layer
        
    def Ind(self, val, list_t):
        pos = 0
        diff = val
        for i, t in enumerate(list_t):
            if (abs(t - val) < diff):
                diff = abs(t - val)
                pos = i
        return pos

    def f(self, teta): return lambda t, z: teta[0] * math.sin(teta[1]*z) + teta[2]
    def da_dt(self, teta): return lambda t, a, z, list_t: (-a.T) * teta[0] * teta[1] * math.cos(teta[1]*z[self.Ind(t, list_t)])
    def df_dteta(self, teta): return lambda t, z: np.array([math.sin(teta[1]*z), teta[0] * z * math.cos(teta[1]*z), 1])
    
def npZero(a, b):
    return np.random.normal(0.0, 0.0, (a, b))

class Lin_ODENet_p2(Solution):
    def __init__(self, h, lr, num_layer = -1):
        self.t0, self.tn = -1.0, 1.0
        self.h = h

        self.t = np.arange(self.t0, self.tn + self.h, self.h)
        self.teta = npZero(2, 2)
        print("default teta = ", self.teta)
        
        self.losses = []
        self.lr, self.lr0 = lr, lr
        self.num_layer = num_layer
    
    def f(self, teta): return lambda t, z: np.dot(z, teta)   
    def da_dt(self, teta): return lambda t, a, z, list_t: np.dot(a.T, teta)
    def df_dteta(self, teta): return lambda t, z: z

class Solution2:
    def __init__(self, f, t0, z0, tn, teta, da_dt, df_dteta, h, target, lr):
        self.f = f
        self.da_dt = da_dt
        self.df_dteta = df_dteta
        self.t0, self.tn = t0, tn
        self.z0 = z0
        self.teta = teta
        self.h = h
        self.losses = []
        self.target = np.array(target)
        self.lr = lr
    
    def get_loss(self):
        self.loss = (self.target - self.zn[-1])**2
        return self.loss
    
    def der_loss_z(self, z, z_true):
        return 2 * abs(z - z_true)

    def forward(self):
        self.t, self.zn = Solver.Get_value(x0 = self.t0, y0 = self.z0, xn = self.tn, f = self.f, teta = self.teta, h=self.h)
        # print("- t", self.t)
        # print("- zn", self.zn)
        
    def backward(self):
        loss = self.get_loss()
        self.losses.append(loss)
        print("LOSS ", loss)
        
        
        ord_loss = int(math.log10(abs(loss)))
        ord_lr = int(math.log10(self.lr))
        if  ord_loss >= 0 and ord_loss - 1 >= abs(ord_lr):
            self.lr /= 10
            print("if1:", self.loss)
            print("if1:", self.lr)
        elif  ord_loss < 0 and ord_loss <= ord_lr:
            self.lr /= 10
            print("if2:", self.loss)
            print("if2:", self.lr)
        elif ord_loss - 2 < ord_lr and ord_loss > 0:
            self.lr *= 10
            print("if3:", self.loss)
            print("if3:", self.lr)

        # print(self.zn.shape, self.target.shape)

        a0 = self.der_loss_z(self.zn[-1], self.target)
        
        self.a, self.z, self.t = Solver.Get_value_a(x0=self.tn, y0=self.z0, xn=self.t0, f=self.f, teta=self.teta, a0=a0, der_a = self.da_dt, h = self.h, xxk=self.t, yyk= self.zn)
 
        if (not np.array_equal(self.z, self.zn)): 
            print("ERRRR")
            print("teta", self.teta)
            print("self.dL_dteta", self.dL_dteta)
            print('self.a', self.a)
            print("z", self.z)
            print("zn", self.zn)
            return

        self.dL_dteta = 0
        f = lambda i: self.a[i] * self.df_dteta(self.t[i], self.z[i], self.teta) * self.h
        for i in range(1, len(self.z) - 1):
            self.dL_dteta += f(i)
        self.dL_dteta += self.h/2 *(f(0) + f(len(self.z) - 1))
        
        self.dL_dteta *= -1
        print("self.dL_dteta", self.dL_dteta)
        print('self.lr', self.lr)

        ord_dl = np.array([int(math.log10(abs(self.dL_dteta[0]))), int(math.log10(abs(self.dL_dteta[1])))])
        if  (ord_dl < 0).all() and (ord_dl <= ord_lr).all():
            self.lr = 0.1 * 10**(ord_dl)
            print("if4_dl:", self.dL_dteta)
            print("if4_lr:", self.lr)

        self.teta = self.teta - self.dL_dteta * self.lr

class Solution_prev_v0:
    def __init__(self, f, t0, z0, tn, teta, da_dt, df_dteta, h, target, lr):
        self.f = f
        self.da_dt = da_dt
        self.df_dteta = df_dteta
        self.t0, self.tn = t0, tn
        self.z0 = z0
        self.teta = teta
        self.h = h
        self.losses = []
        self.target = np.array(target)
        self.lr = lr
    
    def get_loss(self):
        self.loss = sum((self.target - self.zn)**2) / len(self.target)
        return self.loss
    
    def der_loss_z(self, z, z_true):
        return 2 / 1 * abs(z [-1] - z_true[-1])

    def forward(self):
        self.t, self.zn = Solver.Get_value(x0 = self.t0, y0 = self.z0, xn = self.tn, f = self.f, teta = self.teta, h=self.h)
        # print("- t", self.t)
        # print("- zn", self.zn)
        
    def backward(self):
        loss = self.get_loss()
        self.losses.append(loss)
        print("LOSS ", loss)
        
        
        ord_loss = int(math.log10(abs(loss)))
        ord_lr = int(math.log10(self.lr))
        if  ord_loss >= 0 and ord_loss - 1 >= abs(ord_lr):
            self.lr /= 10
            print("if1:", self.loss)
            print("if1:", self.lr)
        elif  ord_loss < 0 and ord_loss <= ord_lr:
            self.lr /= 10
            print("if2:", self.loss)
            print("if2:", self.lr)
        elif ord_loss - 2 < ord_lr and ord_loss > 0:
            self.lr *= 10
            print("if3:", self.loss)
            print("if3:", self.lr)

        # print(self.zn.shape, self.target.shape)

        a0 = self.der_loss_z(self.zn, self.target)
        
        self.a, self.z, self.t = Solver.Get_value_a(x0=self.tn, y0=self.z0, xn=self.t0, f=self.f, teta=self.teta, a0=a0, der_a = self.da_dt, h = self.h, xxk=self.t, yyk= self.zn)
 
        if (not np.array_equal(self.z, self.zn)): 
            print("ERRRR")
            print("teta", self.teta)
            print("self.dL_dteta", self.dL_dteta)
            print('self.a', self.a)
            print("z", self.z)
            print("zn", self.zn)
            return

        self.dL_dteta = 0
        f = lambda i: self.a[i] * self.df_dteta(self.t[i], self.z[i], self.teta) * self.h
        for i in range(1, len(self.z) - 1):
            self.dL_dteta += f(i)
        self.dL_dteta += self.h/2 *(f(0) + f(len(self.z) - 1))
        
        self.dL_dteta *= -1
        print("self.dL_dteta", self.dL_dteta)
        print('self.lr', self.lr)

        ord_dl = int(math.log10(abs(self.dL_dteta)))
        if  ord_dl < 0 and ord_dl <= ord_lr:
            self.lr = 0.1 * 10**(ord_dl)
            print("if4_dl:", self.dL_dteta)
            print("if4_lr:", self.lr)

        self.teta = self.teta - self.dL_dteta * self.lr

def ode_solve(z0, t0, t1, f):
    h_max = _h_
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())

    h = (t1 - t0) / n_steps
    t = t0
    z = z0
    z_list = [z0]
    global mt
    for i_step in range(n_steps):
        if mt == metods[1]:
            z_p = z + h * f(z, t)
            z = z + (f(z, t) + f(z_p, t + h)) * h / 2
            z_list += [z]
            t = t + h
        elif mt == metods[2]:
            k1 = f(z, t)
            k2 = f(z + k1 * h / 2, t + h / 2)
            k3 = f(z + k2 * h / 2, t + h / 2)
            k4 = f(z + k3 * h, t + h)
            z = z + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
            z_list += [z]
            t = t + h
        else:
            z = z + h * f(z, t)
            z_list += [z]
            t = t + h
        
    # global _plot_
    # global _z_t_
    # if _plot_:    #     _z_t_ += [[plot_z_t(z_list, t0, h)]]
    #     _plot_ = False

    return z

class Function_for_ode(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        out = self.forward(z, t)

        n_batch, dim_z = z.size()

        a = grad_outputs
        dfdz, dfdt, *dfdp = torch.autograd.grad(
            (out,),
            (z, t) + tuple(self.parameters()),
            grad_outputs=(a),
            allow_unused=True,
            retain_graph=True,
        )
        if dfdp is not None:
            dfdp = torch.cat([p_grad.flatten() for p_grad in dfdp])
            dfdp = dfdp.expand(n_batch, -1) / n_batch

        return out, dfdz, dfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class Matrix_func(Function_for_ode):
    def __init__(self, in_dim, hid_dim):
        super(Matrix_func, self).__init__()

        print("param = ", in_dim * hid_dim + hid_dim**2 + hid_dim * hid_dim)

        self.lin1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.lin2 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin3 = nn.Linear(hid_dim, in_dim, bias=True)
        self.f1 = nn.ReLU()
        self.f2 = nn.Sigmoid()
        self.f3 = nn.Tanh()

    def forward(self, x, t):
        h = self.lin1(x).to(x)
        h = self.f3(self.lin2(h)).to(x)
        out = self.lin3(h).to(x)
        return out

class Matrix_func1(Function_for_ode):
    def __init__(self, in_dim, hid_dim):
        super(Matrix_func1, self).__init__()

        # print("param = ", in_dim * hid_dim + hid_dim**2 + hid_dim * in_dim)

        self.lin1 = nn.Linear(in_dim, 2, bias=False)
        self.lin2 = nn.Linear(2, hid_dim, bias=False)
        self.lin3 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin4 = nn.Linear(hid_dim, 2, bias=True)
        self.lin5 = nn.Linear(2, in_dim, bias=True)
        self.f1 = nn.ReLU()
        self.f2 = nn.Sigmoid()
        self.f3 = nn.Tanh()

    def forward(self, x, t):
        h =  self.f3(self.lin1(x)).to(x)
        h =  self.f3(self.lin2(h)).to(x)
        h = self.f3(self.lin3(h)).to(x)
        h =  self.f3(self.lin4(h)).to(x)
        out = self.lin5(h).to(x)
        return out

class ODEAdjoint(tr.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, Function_for_ode)

        n_t = t.size(0)
        n_batch, dim_z = z0.size()

        with torch.no_grad():
            z = torch.zeros(n_t, n_batch, dim_z).to(z0)
            z[0] = z0
            for i in range(n_t - 1):
                z0 = ode_solve(z0, t[i], t[i + 1], func)
                z[i + 1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors

        n_t = t.size(0)
        dim_t, n_batch, dim_z = z.size()
        n_p = len(flat_parameters)

        def augmented_dynamics(aug, t_i):
            z_i = aug[:, :dim_z].view(n_batch, dim_z)
            a_z = aug[:, dim_z : 2 * dim_z].view(n_batch, dim_z)

            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                f_val, adfdz, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a_z)

                adfdz = (
                    adfdz.to(z_i)
                    if adfdz is not None
                    else torch.zeros(n_batch, dim_z).to(z_i)
                )
                adfdp = (
                    adfdp.to(z_i)
                    if adfdp is not None
                    else torch.zeros(n_batch, n_p).to(z_i)
                )

            return torch.cat((f_val, -adfdz, -adfdp), dim=1)

        with torch.no_grad():
            a_z = torch.zeros(n_batch, dim_z).to(dLdz)
            a_p = torch.zeros(n_batch, n_p).to(dLdz)
            a_t = torch.zeros(n_batch, n_t, dim_t).to(dLdz)

            for i in range(n_t - 1, 0, -1):
                a_z += dLdz[i]
                s0 = torch.cat((z[i], a_z, torch.zeros(n_batch, n_p).to(z)), dim=-1)
                aug_ans = ode_solve(s0, t[i], t[i - 1], augmented_dynamics)
                a_z = aug_ans[:, dim_z : 2 * dim_z]
                a_p += aug_ans[:, 2 * dim_z :]
                del s0, aug_ans
            a_z += dLdz[0]
        return a_z, a_t, a_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, Function_for_ode)
        self.func = func

    def forward(self, z0, t=tr.Tensor([0.0, 1.0]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)

        if return_whole_sequence:
            return z
        else:
            return z[-1]

    






















