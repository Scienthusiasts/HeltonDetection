import torch
import torch.nn as nn


def clip_by_tensor(t, t_min, t_max):
    # 对预测结果做数值截断
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred, target):
    return torch.pow(pred - target, 2)

def BCELoss(pred, target, epsilon=1e-7):
    pred    = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def FocalLoss(pred, target, loss_fcn, gamma=1.5, alpha=0.25):
    loss = loss_fcn(pred, target)
    # p_t = torch.exp(-loss)
    # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

    # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    pred_prob = torch.sigmoid(pred)  # prob from logits
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= alpha_factor * modulating_factor
    return loss


def QFocalLoss(pred, target, loss_fcn, gamma=1.5, alpha=0.25):
    loss = loss_fcn(pred, target)

    pred_prob = torch.sigmoid(pred)  # prob from logits
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.abs(target - pred_prob) ** gamma
    loss *= alpha_factor * modulating_factor
    return loss


def GIoULoss(box_giou, target):
    # 定位损失(直接用的giou)
    loss = (1 - box_giou)[target == 1]
    return torch.mean(loss)





class Loss(nn.Module):
    def __init__(self, loss_type:str, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        if self.loss_type == 'BCELoss':
            loss = BCELoss(pred, target)
        elif self.loss_type == 'MSELoss':
            loss = MSELoss(pred, target)
        elif self.loss_type == 'FocalLoss':
            loss = FocalLoss(pred, target, self.loss_fcn, self.gamma, self.alpha)
        elif self.loss_type == 'QFocalLoss':
            loss = QFocalLoss(pred, target, self.loss_fcn, self.gamma, self.alpha)
        elif self.loss_type == 'GIoULoss':
            loss = GIoULoss(pred, target)
        
        return torch.mean(loss)

