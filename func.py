import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.integrate import solve_ivp
# from jax.experimental.ode import odeint
from scipy.integrate import odeint

colors = sns.color_palette('turbo', 3) 
class Circle:
    def __init__(self, r, x0 = 0, y0 = 0):
        self.x0 = x0
        self.y0 = y0
        self.r = r
    
    def Get_points(self, d_phi = math.pi / 20):
        t = np.arange(0, math.pi * 2, d_phi)
        x = self.r * np.cos(t)
        y = self.r * np.sin(t)
        return self.Shift((x, y))    
    def Shift(self, points):
        return np.array((points[0] + self.x0, points[1] + self.y0))

class Xy:
    def __init__(self):
        pass
    @classmethod
    def Get_points(self, d_x = 0.1, x0 = -1, xn=1):
        x = np.arange(x0, xn, d_x)
        y = x / (3 * x + 4)**3
        return np.array((x, y))
    
    @classmethod
    def Get_der(self):
        return lambda x: (4 - 6*x) / (3*x + 4)**4

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
        if len(self.z) != 21:
            print("len(self.z) = ", len(self.z))
            print("len(t_for_z) = ", len(t_for_z))
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
##1
class Lin_Layer(Solution):
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
##2
class Sq2_Layer(Solution):

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

class Sq3_Layer(Solution):
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

# class Sq4_Layer(Solution):
#     def __init__(self, h, lr, num_layer = -1):

#         self.f = lambda t, z, teta: teta[4] * z**4 + teta[3] * z**3 + teta[2] * z**2 + teta[1] * z + teta[0]
#         self.da_dt = lambda t, a, z, teta: (-a.T) * (teta[4] * 4 * z**3 + teta[3] * 3 * z**2 + teta[2] * 2 *z + teta[1])
#         self.df_dteta = lambda t, z, teta: np.array([1, z, z**2, z**3, z**4])

#         self.t0, self.tn = 0.0, 1.0

#         self.teta = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
#         self.h = h
#         self.losses = []
#         self.lr = lr
#         self.lr0 = lr
#         self.num_layer = num_layer

# class Sq5_Layer(Solution):
#     def __init__(self, h, lr, num_layer = -1):

#         self.f = lambda t, z, teta: teta[5] * z**5 + teta[4] * z**4 + teta[3] * z**3 + teta[2] * z**2 + teta[1] * z + teta[0]
#         self.da_dt = lambda t, a, z, teta: (-a.T) * (teta[5] * 4 * z**4 + teta[4] * 4 * z**3 + teta[3] * 3 * z**2 + teta[2] * 2 *z + teta[1])
#         self.df_dteta = lambda t, z, teta: np.array([1, z, z**2, z**3, z**4, z**5])

#         self.t0, self.tn = 0.0, 1.0

#         self.teta = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#         self.h = h
#         self.losses = []
#         self.lr = lr
#         self.lr0 = lr
#         self.num_layer = num_layer
# ##3
class Exp_Layer(Solution):
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

# class Sin_Layer(Solution):
#     def __init__(self, h, lr, num_layer = -1):
        
#         self.f = lambda t, z, teta: teta[0] * math.sin(teta[1]*z) + teta[2]
#         self.da_dt = lambda t, a, z, teta: (-a.T) * teta[0] * teta[1] * math.cos(teta[1]*z)
#         self.df_dteta = lambda t, z, teta: np.array([math.sin(teta[1]*z), teta[0] * z * math.cos(teta[1]*z), 1])

#         self.t0, self.tn = 0.0, 1.0
#         self.teta = np.array([0.1, 0.1, 0.1])
#         self.h = h
#         self.losses = []
#         self.lr = lr
#         self.lr0 = lr
#         self.num_layer = num_layer
    
def npZero(a, b):
    return np.random.normal(0.0, 0.0, (a, b))



class Lin_Layer_p2(Solution):
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

    