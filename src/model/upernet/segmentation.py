import torch
import torch.nn as nn

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit

    def forward(self, img_data, label, *, segSize=None):
        # training
        if segSize is None:
            pred = self.decoder(self.encoder(img_data))
            label=torch.squeeze(label)
            loss = self.crit(pred, label)
            acc = self.pixel_acc(pred, label)
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(img_data), segSize=segSize)
            return pred
