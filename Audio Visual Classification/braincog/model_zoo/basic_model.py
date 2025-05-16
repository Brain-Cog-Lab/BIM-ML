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
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, IdenticalFusion, AvattnlFusion, OGMGE_MetamodalFusion


class AVClassifier(BaseModule):
    def __init__(self, num_classes=20, step=15, node_type=LIFNode, encode_type='direct', *args, **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.fusion = kwargs['fusion_method'] if 'fusion_method' in kwargs else False
        self.modality = kwargs['modality'] if 'modality' in kwargs else False
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
        elif self.fusion == 'avattn':
            self.fusion_module = AvattnlFusion(output_dim=n_classes)
        elif self.fusion == 'ogmge_metamodal':
            self.fusion_module = OGMGE_MetamodalFusion(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))

        self.audio_net = resnet18(node_type=ReLUNode, step=1, dataset=kwargs["dataset"], modality='audio')
        self.visual_net = resnet18(node_type=ReLUNode, step=1, dataset=kwargs["dataset"], modality='visual')

        self.audio_fc = nn.Linear(512, n_classes)
        self.visual_fc = nn.Linear(512, n_classes)

    def forward(self, input):
        """
            audio: shape [128, t, 1, 128, 128]
            visual: shape [128, t, 3, 128, 128]
        """
        if self.modality == "audio":
            audio, visual = input, None
        elif self.modality == "visual":
            audio, visual = None, input
        else:
            audio, visual = input

        withSampling = False

        if visual is not None:
            if len(visual.size()) == 6:
                (B, T, C, N, H, W) = visual.size()
                visual = visual.permute(0, 3, 1, 2, 4, 5).contiguous()
                visual = visual.view(B * N, T, C, H, W)
                withSampling = True


        if self.modality == "audio":
            audio = self.encoder(audio)
        elif self.modality == "visual":
            visual = self.encoder(visual)
        else:
            audio, visual = self.encoder(audio), self.encoder(visual)
        self.reset()

        output_a_list, output_v_list, disc_pred_a_list, disc_pred_v_list, out_list = [], [], [], [], []
        step = self.step

        if self.modality == "audio":
            for t in range(step):
                a = audio[t]
                a = self.audio_net(a)

                a = F.adaptive_avg_pool2d(a, 1)
                a = torch.flatten(a, 1)

                output_a = self.audio_fc(a)

                output_a_list.append(output_a)

            return None, None, sum(output_a_list) / len(output_a_list)

        if self.modality == "visual":
            for t in range(step):
                v = visual[t]
                v = self.visual_net(v)

                (_, C, H, W) = v.size()
                if withSampling:
                    B = v.size()[0] // 3
                else:
                    B = v.size()[0]
                v = v.view(B, -1, C, H, W)
                v = torch.mean(v, dim=1)

                v = F.adaptive_avg_pool2d(v, 1)
                v = torch.flatten(v, 1)

                output_v = self.visual_fc(v)

                output_v_list.append(output_v)

            return None, None, sum(output_v_list) / len(output_v_list)

        if self.modality == "audio-visual":
            for t in range(step):
                a, v = audio[t], visual[t]

                a = self.audio_net(a)
                v = self.visual_net(v)

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = torch.mean(v, dim=1)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool2d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)
                output_a, output_v, out = self.fusion_module(a, v)
                # for test flops only
                # output_a = self.audio_fc(output_a)
                # output_v = self.visual_fc(output_v)
                output_a_list.append(output_a), output_v_list.append(output_v), out_list.append(out)

            return sum(output_a_list) / len(output_a_list), sum(output_v_list) / len(output_v_list), sum(out_list) / len(out_list)


@register_model
def AVresnet18(**kwargs):
    model = AVClassifier(**kwargs)
    return model