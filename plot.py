import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import torch as tr

def make_batch(x_val, y_val, num):
    ind1 = 0
    ind2 = num
    l = []
    for i in range(0, len(x_val) // num):
        if ind2 > len(x_val): break
        if ind1 + 1 == ind2:
            x = x_val[ind1:ind2].view(1, len(x_val[0]))
            y = y_val[ind1:ind2].view(1, len(y_val[0]))
            l += [(x, y)]
        else:
            x = x_val[ind1:ind2]
            y = y_val[ind1:ind2]
            l += [(x, y)]
            
        ind1 = ind2
        ind2 = ind1 + num
    if num >= len(x_val):
      x = x_val[:len(x_val)]
      y = y_val[:len(x_val)]
      l += [(x, y)]
    elif num != 1:
      l += [(x_val[len(x_val) - num:len(x_val)], y_val[len(x_val) - num:len(x_val)])]
    return l

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

def plot_res1(x_values, y_values, t_values, l1, device, ep, list_loss, z0, num):
    zz = []
    tt = []
    # pred_val_x = tr.zeros(len(x_values), 1).type_as(z0)
    # pred_val_y = tr.zeros(len(x_values), 1).type_as(z0)
    pred_val_x = list()
    pred_val_y = list()
    val_y = list()
    list_batch = make_batch(x_values, y_values, num=1)
  
    for ind, batch in enumerate(list_batch):
        x, y = batch
        x_, y_ = x.to(device), y.to(device)
        t_ = t_values.to(device)
        z_ = l1(x_, t_, return_whole_sequence=True)
        zz += [z_.cpu()]
        tt += [t_values]
        # print(z_[:, :,  0])
        
        
        pred_val_x += z_[-1][:, 0].flatten().tolist()
        pred_val_y += z_[-1][:, 1].flatten().tolist()
        val_y += y_[:, 1].flatten().tolist()


    plt.figure(figsize=(16, 8))
    plt.subplot(221)
    k = len(zz) // 50 + 1
    for z_i, t_i in zip(zz[::k], tt[::k]):
        x_np = to_np(t_i)
        y_np = to_np(z_i)
        plt.plot(x_np[:, 0], y_np[:, 0], color = 'blue')
    plt.xlabel('t')
    plt.ylabel('z(t)')

    plt.subplot(222)
    x_np = pred_val_x
    y_np = val_y
    plt.plot(x_np, y_np, color = 'green', label  = 'true')
    
    y_np = pred_val_y
    plt.plot(x_np, y_np, color = 'red', label  = 'pred') 
    plt.xlabel('x')
    plt.ylabel('y')


    plt.subplot(223)
    plt.plot(np.arange(0, len(list_loss)), list_loss, color = 'green', label  = 'loss')
    plt.xlabel('ep')
    plt.ylabel('loss')


    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    x_np = pred_val_x
    y_np = val_y
    plt.scatter(x_np, y_np, color = 'green', label  = 'true')
    
    y_np = pred_val_y
    plt.scatter(x_np, y_np, color = 'red', label  = 'pred') 
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(122)
    x_np = pred_val_x
    y_np = val_y
    plt.scatter(x_np, y_np, color = 'green', label  = 'true')
    
    y_np = pred_val_y
    plt.plot(x_np, y_np, color = 'red', label  = 'pred') 
    plt.xlabel('x')
    plt.ylabel('y') 

def plot_z_t(z_list, t0, h):
    print(len(z_list))
    z_list = tr.cat(z_list, dim = 1)
    tz = []
    for i in z_list:
        z = i[::2].tolist()
        t = np.arange(t0, t0 + h * len(z), h)
        tz += [(t, z)]
    return np.array(tz)




def plot_loss(list_loss, mt = []):
    fig, ax = plt.subplots(1, figsize= (10, 5))
    ep = np.arange(0, len(list_loss[0]))
    
    for i in range(len(mt)):
        ax.plot(ep, list_loss[i], label  = 'loss' + " " + mt[i])
        
    ax.set(xlabel='ep', ylabel='loss')
    plt.legend()
    plt.show()
    
def plot_result(x_values, y_values, t_values, l1, func_run):
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    fl = True

    x_val = tr.Tensor(x_values.tolist())
    y_val = tr.Tensor(y_values.tolist())
    pred = func_run(x_val, y_val, t_values, l1, train=False)
    
    list_x = x_values.numpy()
    list_y = y_values.numpy()
    list_pred = pred.detach().numpy()
    
    if fl: 
        ax[0, 0].plot(list_x, list_y, color = 'g', label = "true function")
        ax[0, 1].scatter(list_x, list_y, color = 'g', label = "true function")
        ax[1, 0].scatter(list_x, list_y, color = 'g', label = "true function")
        ax[1, 1].scatter(list_x, list_y, color = 'g', label = "true function")
        fl = False
    ax[0, 0].plot(list_x, list_pred)
    
    ax[0, 1].scatter(list_x, list_pred)
    ax[1, 0].plot(list_x, list_pred)
    
    ax[0, 0].set(xlabel='x', ylabel='y')
    ax[0, 1].set(xlabel='x', ylabel='y')
    ax[1, 0].set(xlabel='x', ylabel='y')
    ax[1, 1].set(xlabel='x', ylabel='y')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()

def plot_result1(x_values, y_values, t_values, l1, func_run, mt):
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    colors = ['r', 'b', 'm']
    fl = True
    for i in range(len(mt)):
        x_val = tr.Tensor(x_values[i].tolist())
        y_val = tr.Tensor(y_values[i].tolist())
        pred = func_run[i](x_val, y_val, t_values[i], l1[i], train=False)
        
        list_x = x_values[i].numpy()
        list_y = y_values[i].numpy()
        list_pred = pred.detach().numpy()
        
        if fl: 
            ax[0, 0].plot(list_x, list_y, color = 'g', label = "true function")
            ax[0, 1].scatter(list_x, list_y, color = 'g', label = "true function")
            ax[1, 0].scatter(list_x, list_y, color = 'g', label = "true function")
            ax[1, 1].scatter(list_x, list_y, color = 'g', label = "true function")
            fl = False
        ax[0, 0].plot(list_x[:, 0], list_pred[:, 0], color = colors[i], label = mt[i])
        ax[0, 1].scatter(list_x[:, 0], list_pred[:, 0], color = colors[i], label = mt[i])
        ax[1, 0].plot(list_x[:, 0], list_pred[:, 0], color = colors[i], label = mt[i])
        ax[0, 0].plot(list_x[:, 1], list_pred[:, 1], color = colors[i])
        ax[0, 1].scatter(list_x[:, 1], list_pred[:, 1], color = colors[i])
        ax[1, 0].plot(list_x[:, 1], list_pred[:, 1], color = colors[i])
    
    ax[0, 0].set(xlabel='x', ylabel='y')
    ax[0, 1].set(xlabel='x', ylabel='y')
    ax[1, 0].set(xlabel='x', ylabel='y')
    ax[1, 1].set(xlabel='x', ylabel='y')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()













def plot_result21(x_values, y_values, t_values, l1, func_run):
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    
    x_val = tr.Tensor(x_values.tolist())
    y_val = tr.Tensor(y_values.tolist())
    pred = func_run(x_val, y_val, t_values, l1, train=False)
    
    list_x_ = x_values.numpy()
    list_y_ = y_values.numpy()
    list_pred_ = pred.detach().numpy()
    
    list_x, list_y, list_pred = [], [], []
    
    for i in range(len(list_x_)):
        list_x += [list_x_[i][0]]
        list_y += [list_y_[i][0]]
        list_pred += [(list_pred_[i][0] + list_pred_[i][1]) / 2]
    
    print(2)
    
    ax[0, 0].plot(list_x, list_y, color = 'g')
    ax[0, 0].plot(list_x, list_pred, color = 'r')
    
    ax[0, 1].scatter(list_x, list_y, color = 'g')
    ax[0, 1].scatter(list_x, list_pred, color = 'r')
    
    plt.legend()
    plt.show()
    
def convert(a):
    return np.hstack([a[:, None]])

def convert_to_tensor(a):
    return tr.from_numpy(a[:, :, None])    
    
def plot_result_din(z0, time, l1, values, func_run):
    
    t = convert_to_tensor(time)
    true_val = values
    pred_val = func_run(z0, z0, t, l1, train=False)
    
    x_true = true_val[:, 0][:, 0]
    y_true = true_val[:, 0][:, 1]
    
    x_pred = pred_val[:, 0][:, 0]
    y_pred = pred_val[:, 0][:, 1]
    
    # x_true = true_val[:, 0]
    # y_true = true_val[:, 0]
    
    # x_pred = pred_val[:, 0]
    # y_pred = pred_val[:, 0]
    
    
    list_x = x_true.detach().numpy()
    list_y = y_true.detach().numpy()
    list_pred_x = x_pred.detach().numpy()
    list_pred_y = y_pred.detach().numpy()
    
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    
    ax[0, 0].plot(list_x, list_y, color = 'g', label = "true function")
    # ax[0, 1].scatter(list_x, list_y, color = 'g', label = "true function")
    ax[1, 0].scatter(list_x, list_y, color = 'g', label = "true function")
    ax[1, 1].scatter(list_x, list_y, color = 'g', label = "true function")

    ax[0, 0].plot(list_pred_x, list_pred_y, label = "pred")
    ax[0, 1].scatter(list_pred_x, list_pred_y, label = "pred")
    ax[1, 0].plot(list_pred_x, list_pred_y, label = "pred")
    
    ax[0, 0].set(xlabel='x', ylabel='y')
    ax[0, 1].set(xlabel='x', ylabel='y')
    ax[1, 0].set(xlabel='x', ylabel='y')
    ax[1, 1].set(xlabel='x', ylabel='y')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()

def plot_result_din2(z0, time, l1, values, func_run, l_batch, model):
    t = convert_to_tensor(time)
    true_val = values
    pred_val = func_run(z0, z0, t, l1, train=False)
    
    x_true = true_val[:, 0][:, 0]
    y_true = true_val[:, 0][:, 1]
    
    x_pred = pred_val[:, 0][:, 0]
    y_pred = pred_val[:, 0][:, 1]
    
    # x_true = true_val[:, 0]
    # y_true = true_val[:, 0]
    
    # x_pred = pred_val[:, 0]
    # y_pred = pred_val[:, 0]
    
    
    
    list_x = x_true.detach().numpy()
    list_y = y_true.detach().numpy()
    list_pred_x = x_pred.detach().numpy()
    list_pred_y = y_pred.detach().numpy()
    
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    
    ax[0, 0].plot(list_x, list_y, color = 'g', label = "true function")
    # ax[0, 1].scatter(list_x, list_y, color = 'g', label = "true function")
    ax[1, 0].scatter(list_x, list_y, color = 'g', label = "true function")
    ax[1, 1].scatter(list_x, list_y, color = 'g', label = "true function")

    ax[0, 0].plot(list_pred_x, list_pred_y, label = "pred")
    ax[0, 1].scatter(list_pred_x, list_pred_y, label = "pred")
    ax[1, 0].plot(list_pred_x, list_pred_y, label = "pred")
    
    for batch in l_batch:
        samples, targets = batch
        outputs = model(samples)
        xx = []
        yy = []
        for o in outputs:
            xx += [o.detach()[0].item()]
            yy += [o.detach()[1].item()]
        ax[1, 1].plot(xx, yy)
    
    ax[0, 0].set(xlabel='x', ylabel='y')
    ax[0, 1].set(xlabel='x', ylabel='y')
    ax[1, 0].set(xlabel='x', ylabel='y')
    ax[1, 1].set(xlabel='x', ylabel='y')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()
    
def plot_result_din_list(z0, time, l1, values, func_run, metods):
    fl = True
    fig, ax =  plt.subplots(2, 2, figsize = (20, 10))
    for i, mt in enumerate(metods):
        t = convert_to_tensor(time[i])
        true_val = values[i]
        pred_val = func_run[i](z0[i], z0[i], t, l1[i], train=False)
        
        x_true = true_val[:, 0][:, 0]
        y_true = true_val[:, 0][:, 1]
        
        x_pred = pred_val[:, 0][:, 0]
        y_pred = pred_val[:, 0][:, 1]
        
        # x_true = true_val[:, 0]
        # y_true = true_val[:, 0]
        
        # x_pred = pred_val[:, 0]
        # y_pred = pred_val[:, 0]
        
        
        list_x = x_true.detach().numpy()
        list_y = y_true.detach().numpy()
        list_pred_x = x_pred.detach().numpy()
        list_pred_y = y_pred.detach().numpy()
        
        
        if fl:
            ax[0, 0].plot(list_x, list_y, color = 'g', label = "true function")
            # ax[0, 1].scatter(list_x, list_y, color = 'g', label = "true function")
            ax[1, 0].scatter(list_x, list_y, color = 'g', label = "true function")
            ax[1, 1].scatter(list_x, list_y, color = 'g', label = "true function")
            fl = False

        ax[0, 0].plot(list_pred_x, list_pred_y, label = "pred " + mt)
        ax[0, 1].scatter(list_pred_x, list_pred_y, label = "pred " + mt)
        ax[1, 0].plot(list_pred_x, list_pred_y, label = "pred " + mt)
    
    ax[0, 0].set(xlabel='x', ylabel='y')
    ax[0, 1].set(xlabel='x', ylabel='y')
    ax[1, 0].set(xlabel='x', ylabel='y')
    ax[1, 1].set(xlabel='x', ylabel='y')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()
    
