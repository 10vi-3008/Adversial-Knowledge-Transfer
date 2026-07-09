import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pdb

def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        logits = model(train_ifgsm_data)
        loss = ce_loss(logits,train_batch_labels.cuda())
        loss.backward()
        train_grad = train_ifgsm_data.grad.detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    return train_ifgsm_data

def rslad_inner_loss(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits


def rslad_inner_loss_xadv(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits, x_adv


def rslad_inner_loss_only_return(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    return x_adv


  
def rslad_inner_loss_attack_xadv(model,
                teacher,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    criterion_kl2 = nn.KLDivLoss(reduction="batchmean")
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.no_grad():
            teacher_logits = teacher(x_natural)
            teacher_logits_adv = teacher(x_adv)
            teacher_diff = torch.abs(teacher_logits - teacher_logits_adv)
        

        with torch.enable_grad():

            nat_logits = model(x_natural)
            adv_logits = model(x_adv)
            student_diff = torch.abs(nat_logits - adv_logits)

            diff_kl_loss = 10 * criterion_kl2(F.log_softmax(student_diff, dim=1), F.softmax(teacher_diff.detach(), dim=1))

            loss_kl = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss = torch.sum(loss_kl) + diff_kl_loss
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits, x_adv

def rslad_inner_loss_xadv_step(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    x_adv_list = []
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv_list.append(Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False))
    model.train()

    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv_list[len(x_adv_list)-1])
    return logits, x_adv_list


def adaad_inner_grad_loss(model,
                     teacher_model, true_label,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()
    import copy
    new_teacher = copy.deepcopy([p.data for p in teacher_model.parameters()])
    new_student = copy.deepcopy([p.data for p in model.parameters()])
    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv



def adaad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv



def adaad_inner_loss2(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        delta = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv - 2 * delta), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def adaad_inner_loss3(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     gamma = 1):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        delta = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl2 = criterion_kl(F.log_softmax(teacher_model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv - 2 * delta), dim=1))
            loss_kl = torch.sum(loss_kl) - gamma *  torch.sum(loss_kl2)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv






def adaad_inner_loss4(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     gamma = 1):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        delta = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl2 = criterion_kl(F.log_softmax(teacher_model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv - 2 * delta), dim=1))
            loss_kl = torch.sum(loss_kl) + gamma *  torch.sum(loss_kl2)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def adaad_inner_loss5(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     gamma = 1):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        delta = x_adv - x_natural
        x_adv.requires_grad_()

        with torch.enable_grad():
            teacher_minus = teacher_model(x_adv - 2* delta)
            student_minus = model(x_adv - 2 * delta)

            student_plus = model(x_adv)
            teacher_plus = teacher_model(x_adv)
            
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax(teacher_plus - teacher_minus, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def rslad_attack(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv
