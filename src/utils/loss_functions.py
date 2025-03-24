import torch
import torch.nn.functional as F
import torchvision.models as models

def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def l2_loss(pred, target):
    return torch.mean((pred - target)**2)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, normalize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.normalize = normalize

    def forward(self, input, target):
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        if self.normalize:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def total_loss(pred, target, vgg_loss_fn, l1_weight=1.0, vgg_weight=0.05):
    l1 = l1_loss(pred, target)
    vgg = vgg_loss_fn(pred, target)
    return l1_weight * l1 + vgg_weight * vgg