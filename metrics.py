import torch
import torch.nn as nn

def dataloss(outputs, labels):
    loss_func = nn.MSELoss()
    return loss_func(outputs, labels)

def pointnetloss(outputs, m3x3, m4x4, m64x64):
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id4x4 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id4x4 = id4x4.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff4x4 = id4x4 - torch.bmm(m4x4, m4x4.transpose(1, 2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return (torch.norm(diff3x3)+torch.norm(diff4x4)+torch.norm(diff64x64)) / float(bs)

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def PDEloss(outputs, inputs):
    u = outputs
    x = inputs
    grad_u = gradients(u, x)[0]
    # calculate first order derivatives
    u_x = grad_u[:, 0]
    u_y = grad_u[:, 1]
    u_z = grad_u[:, 2]
    u_x = u_x.reshape(-1, 1)
    u_y = u_y.reshape(-1, 1)
    u_z = u_z.reshape(-1, 1)
    # calculate second order derivatives
    grad_u_x = gradients(u_x, x)[0]
    u_xx = grad_u_x[:, 0]
    grad_u_y = gradients(u_y, x)[0]
    u_yy = grad_u_y[:, 1]
    grad_u_z = gradients(u_z, x)[0]
    u_zz = grad_u_z[:, 2]

    # reshape for correct behavior of the optimizer

    u_xx = u_xx.reshape(-1, 1)
    u_yy = u_yy.reshape(-1, 1)
    u_zz = u_zz.reshape(-1, 1)

    f = u_xx + u_yy + u_zz
    return (f**2).mean()

def PIPNloss(outputs, m3x3, m4x4, m64x64, inputs, labels,alpha = 1000, beta = 1000000):
    loss = dataloss(outputs, labels)+ alpha*pointnetloss(outputs, m3x3, m4x4, m64x64) + beta * PDEloss(outputs, inputs)
    return loss