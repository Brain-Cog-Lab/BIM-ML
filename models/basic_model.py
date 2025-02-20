import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, IdenticalFusion, MetamodalFusion, OGMGE_MetamodalFusion


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif self.fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif self.fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif self.fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        elif self.fusion == 'identical':
            self.fusion_module = IdenticalFusion(output_dim=n_classes)
        elif self.fusion == 'metamodal':
            self.fusion_module = MetamodalFusion(output_dim=n_classes)
        elif self.fusion == 'ogmge_metamodal':
            self.fusion_module = OGMGE_MetamodalFusion(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual):
        """
            audio: shape [64, 1, 257, 188]
            visual: shape [64, 3, 1, 224, 224]
        """
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if self.fusion != "ogmge_metamodal":
            a, v, out = self.fusion_module(a, v)
            return a, v, out
        else:
            output_a, output_v, disc_pred_a, disc_pred_v, out = self.fusion_module(a, v)
            return output_a, output_v, disc_pred_a, disc_pred_v, out