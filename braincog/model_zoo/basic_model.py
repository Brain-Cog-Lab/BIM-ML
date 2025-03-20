import os
import sys
from functools import partial
from timm.models import register_model
from timm.models.layers import trunc_normal_, DropPath
from braincog.model_zoo.base_module import *
from braincog.model_zoo.resnet import *
from braincog.base.node.node import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, IdenticalFusion, MetamodalFusion, OGMGE_MetamodalFusion


class AVClassifier(BaseModule):
    def __init__(self, num_classes=20, step=15, node_type=LIFNode, encode_type='direct', *args, **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.fusion = kwargs['fusion_method'] if 'fusion_method' in kwargs else False
        n_classes = num_classes

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

        self.audio_net = resnet18(node_type=ReLUNode, step=1, dataset=kwargs["dataset"], modality='audio')
        self.visual_net = resnet18(node_type=ReLUNode, step=1, dataset=kwargs["dataset"], modality='visual')
        self.meta_modality = False

    def forward(self, input, alpha):
        """
            audio: shape [128, t, 1, 128, 128]
            visual: shape [128, t, 3, 128, 128]
        """
        audio, visual = input

        if len(visual.size()) == 6:
            (B, T, C, N, H, W) = visual.size()
            visual = visual.permute(0, 3, 1, 2, 4, 5).contiguous()
            visual = visual.view(B * N, T, C, H, W)

        audio, visual = self.encoder(audio), self.encoder(visual)
        self.reset()

        output_a_list, output_v_list, disc_pred_a_list, disc_pred_v_list, out_list = [], [], [], [], []
        step = self.step

        for t in range(step):
            a, v = audio[t], visual[t]

            a = self.audio_net(a)
            v = self.visual_net(v)

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = torch.mean(v, dim=1)

            if "metamodal" in self.fusion:
                a = a.view(B, -1, C)
                v = v.view(B, -1, C)
                output_a, output_v, disc_pred_a, disc_pred_v, out = self.fusion_module(a, v, alpha)
                output_a_list.append(output_a), output_v_list.append(output_v), disc_pred_a_list.append(disc_pred_a), disc_pred_v_list.append(disc_pred_v), out_list.append(out)
            else:
                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool2d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)
                output_a, output_v, out = self.fusion_module(a, v)
                output_a_list.append(output_a), output_v_list.append(output_v), out_list.append(out)

        if "metamodal" in self.fusion:
            return sum(output_a_list) / len(output_a_list), sum(output_v_list) / len(output_v_list), sum(disc_pred_a_list) / len(disc_pred_a_list), sum(disc_pred_v_list) / len(disc_pred_v_list), sum(out_list) / len(out_list)
        else:
            return sum(output_a_list) / len(output_a_list), sum(output_v_list) / len(output_v_list), sum(out_list) / len(out_list)


@register_model
def AVresnet18(**kwargs):
    model = AVClassifier(**kwargs)
    return model