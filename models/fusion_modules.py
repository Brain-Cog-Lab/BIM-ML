import torch
import torch.nn as nn
from copy import deepcopy

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class IdenticalFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(IdenticalFusion, self).__init__()
        self.fc_a = nn.Linear(input_dim, output_dim)
        self.fc_v = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        x = self.fc_a(x)
        y = self.fc_v(y)
        return x, y, None


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class MetamodalFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(MetamodalFusion, self).__init__()
        self.fc_meta = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(2 * input_dim, output_dim)
        self.discriminator_fc = nn.Linear(input_dim, 1)
    def forward(self, a, v):
        output_a = self.fc_meta(a)
        output_v = self.fc_meta(v)

        output_a_reversed = GradientReversalFunction.apply(output_a)
        output_v_reversed = GradientReversalFunction.apply(output_v)
        disc_pred_a = torch.sigmoid(self.discriminator_fc(output_a_reversed))
        disc_pred_v = torch.sigmoid(self.discriminator_fc(output_v_reversed))
        output = torch.cat((output_a, output_v), dim=1)
        output = self.fc_out(output)
        return disc_pred_a, disc_pred_v, output


class OGMGE_MetamodalFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(OGMGE_MetamodalFusion, self).__init__()
        self.fc_meta = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(2 * input_dim, output_dim)
        self.discriminator_fc = nn.Linear(input_dim, 1)
    def forward(self, a, v):
        output_a = self.fc_meta(a)
        output_v = self.fc_meta(v)

        # meta 部分
        output_a_reversed = GradientReversalFunction.apply(output_a)
        output_v_reversed = GradientReversalFunction.apply(output_v)
        disc_pred_a = torch.sigmoid(self.discriminator_fc(output_a_reversed))
        disc_pred_v = torch.sigmoid(self.discriminator_fc(output_v_reversed))
        output = torch.cat((output_a, output_v), dim=1)
        output = self.fc_out(output)

        # inverse 部分, 需要将其中的一个模态设置为零来观察另外一个模态
        # output_a = deepcopy(output_a)
        # output_v = deepcopy(output_v)
        output_a = self.fc_out(torch.cat((output_a, torch.zeros_like(output_a)), dim=1)).detach()
        output_v = self.fc_out(torch.cat((torch.zeros_like(output_v), output_v), dim=1)).detach()
        return output_a, output_v, disc_pred_a, disc_pred_v, output



# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim=2048):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output

