import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


def GN(images, eps=8/255):



    images = images.clone().detach().cuda()

    adv_images = images + eps*torch.randn_like(images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images


def FGSM(images, labels, model, eps=8/255, random_start=False):
    model.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    adv_images.requires_grad = True
    outputs = model(adv_images)

    cost = loss(outputs, labels)

    grad = torch.autograd.grad(cost, adv_images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + eps*grad.sign()
    delta = torch.clamp(adv_images - images, min= -eps, max= eps)
    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def ReverseFGSM(images, labels, model, eps=8/255, random_start=False):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    adv_images.requires_grad = True
    outputs = model(adv_images)

    cost = - loss(outputs, labels)

    grad = torch.autograd.grad(cost, adv_images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + eps*grad.sign()
    delta = torch.clamp(adv_images - images, min= -eps, max= eps)
    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def softXEnt (input, target):
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]


def PGD(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.eval()

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images

def reversePGD(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = - loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def rslad_like_inner(images, labels, model, teacher_logits, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss_kl = criterion_kl(F.log_softmax(outputs, dim=1),
                                    F.softmax(teacher_logits, dim=1))
        grad = torch.autograd.grad(loss_kl, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images

def rslad_like_inner2(images, labels, model, teacher_logits, teacher_logits_diff, gamma, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        delta = adv_images - images
        student_plus = model(adv_images)
        student_minus = model(images - delta)
        loss_kl2 = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax((teacher_logits_diff).detach(), dim=1))
        loss_kl = criterion_kl(F.log_softmax(student_plus, dim=1),
                                    F.softmax(teacher_logits, dim=1))
        grad = torch.autograd.grad(loss_kl + gamma * loss_kl2, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images

def IGDM_inner(images, labels, model, teacher, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad_()

        with torch.no_grad():
            delta = adv_images - images
            teacher_plus = teacher(images + delta)
            teacher_minus = teacher(images - delta)

        student_plus = model(adv_images)
        student_minus = model(adv_images - 2 * delta)
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax((teacher_plus - teacher_minus).detach(), dim=1))
        grad = torch.autograd.grad(loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def IGDM_inner2(images, labels, model, teacher, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad_()

        with torch.no_grad():
            delta = adv_images - images
            teacher_plus = teacher(images + delta)
            teacher_minus = teacher(images - delta)

        student_plus = model(adv_images)
        student_minus = model(adv_images - 2 * delta)
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax((teacher_plus - teacher_minus).detach(), dim=1))
        outputs = model(adv_images)
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost + loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def IGDM_inner3(images, labels, model, teacher, gamma, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad_()

        with torch.no_grad():
            delta = adv_images - images
            teacher_plus = teacher(images + delta)
            teacher_minus = teacher(images - delta)

        student_plus = model(adv_images)
        student_minus = model(adv_images - 2 * delta)
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax((teacher_plus - teacher_minus).detach(), dim=1))
        outputs = model(adv_images)
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost + gamma * loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def IGDM_inner4(images, labels, model, teacher, gamma, epoch_rate, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad_()

        with torch.no_grad():
            delta = adv_images - images



        with torch.enable_grad():
            teacher_minus = teacher(adv_images - 2* delta)
            student_minus = model(adv_images - 2 * delta)

            student_plus = model(adv_images)
            teacher_plus = teacher(adv_images)
            
            loss_kl = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), F.softmax(teacher_plus - teacher_minus, dim=1))
        outputs = model(adv_images)
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost + gamma*epoch_rate* loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images

def gc_proto_PGD(images, labels, model, proto_logit_diff_tensor, adap, diff_alpha, cur_epoch, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()
    criterion_kl = nn.KLDivLoss(reduction='sum')
    none_kl = nn.KLDivLoss(reduction='none')

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    CE_loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
    for _ in range(steps):
        adv_images.requires_grad = True
        with torch.no_grad():
            delta = adv_images - images
        outputs_minus = model(adv_images - 2 * delta) 
        outputs = model(adv_images)
        cost = CE_loss(outputs, labels)

        adv_probs = F.softmax(outputs, dim=1)
        true_probs = torch.gather(adv_probs, 1, (labels.unsqueeze(1)).long()).squeeze()

        diff_loss = torch.sum(torch.sum(
                    none_kl(F.log_softmax(outputs - outputs_minus, dim=1), 
                            F.softmax((proto_logit_diff_tensor[labels.cpu().detach().numpy()].detach()), dim=1)), dim=1) * (0.0000001 + true_probs) * (0.0000001 + true_probs))
        if adap == "adap":
            loss = cost  + diff_alpha * (cur_epoch/200) * diff_loss
        elif adap == "adap2":
            loss = cost  + diff_alpha * (cur_epoch/200) * (cur_epoch/200) * diff_loss
        elif adap == "adap3":
            if cur_epoch <= 100:
                loss = cost  + diff_alpha * (cur_epoch/200) * 0.1 * diff_loss
            elif cur_epoch <= 150:
                loss = cost  + diff_alpha * ((cur_epoch/200) * 1.2 - 0.5) * diff_loss
            else:
                loss = cost  + diff_alpha * ((cur_epoch/200) * 2.4 - 1.4)  * diff_loss
        else:
            loss = cost + diff_alpha * diff_loss



        grad = torch.autograd.grad(loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def PGD_minus_eps(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def PGD_multiout(images, labels, model, eps=8/255, alpha=2/225, steps=2, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    list_adv_images = []
    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        list_adv_images.append(adv_images)
    model.train()

    return list_adv_images



def TRADES(images, labels, model, eps=8/255, alpha=2/225, steps=10):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    logit_ori = model(images).detach()

    loss = nn.KLDivLoss(reduction='sum')

    adv_images = images.clone().detach()
    adv_images = adv_images + 0.001*torch.randn_like(adv_images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        logit_adv = model(adv_images)
        cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images




############ for batch augmentation #######################################################################################
def PGD_targetattack(images, labels, num_class, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        if steps % 2 == 0:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        adv_images.requires_grad = True
        outputs = model(adv_images)[:,:num_class]
    
        cost = -loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images
###################################### batch augmentation end ###########################################################



def PGD_softlabels(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
    
        cost = softXEnt(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def PGD_var_eps(images, labels, model, eps, alpha, steps=10, random_start=True):
    
    if eps.shape[0] != images.shape[0]:
        raise NotImplementedError('PGD var eps : eps tensor shape error')
    if alpha.shape[0] != images.shape[0]:
        raise NotImplementedError('PGD var eps : alpha tensor shape error')

    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    view_tuple = (-1,) + (1,) * (images.dim()-1) #(-1, 1, 1, 1)
    if random_start:
        adv_images = adv_images + (2 * eps.view(view_tuple) * torch.rand_like(adv_images) - eps.view(view_tuple)) 
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
    
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha.view(view_tuple) *grad.sign()

        delta = torch.max(torch.min(adv_images - images, eps.view(view_tuple)), -eps.view(view_tuple))
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images