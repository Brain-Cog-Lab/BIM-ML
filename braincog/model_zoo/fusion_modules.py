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
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None



class Attention(nn.Module):
    def __init__(self, input_dim=512):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = 4

        self.softmax = nn.Softmax(dim=-1)

        self.query = nn.Linear(self.input_dim, self.input_dim//self.latent_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim//self.latent_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.input_dim//self.latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.input_dim // self.latent_dim, self.input_dim),
        # )


    def forward(self, a, v):
        # ============================== Self Attention =======================================
        query_feature = self.query(a)  # [B, C]
        key_feature = self.key(v)
        attention = self.softmax(torch.matmul(query_feature, key_feature.permute(0, 2, 1)))
        video_value_feature = self.value(v)
        att_output = torch.matmul(attention, video_value_feature)
        output = att_output
        return output


class AVattention(nn.Module):
    def __init__(self, channel=512, av_attn_channel=64):
        super().__init__()
        self.d = av_attn_channel
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(self.d, channel))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
            x, y: [T, B, C]
            return [B, C]
        """
        b, c = x.size()

        ### fuse
        U = x + y

        ### reduction channel
        Z = self.fc(U)  # B, d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(b, c))  # b, c
        attention_weights = torch.stack(weights, 0)  # k,b,c
        attention_weights = self.sigmoid(attention_weights)  # k,bs,channel

        ### fuse
        V = attention_weights[0] * x + attention_weights[1] * y
        return V

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=512, output_dim=100, num_experts=4):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # Define expert networks for both modalities (visual and auditory)
        self.singlemodal_expert = Attention(input_dim)  # Experts for auditory
        # self.multimodal_expert = Attention(input_dim)  # Experts for auditory

        # # Define router for multimodal fusion
        # self.router = nn.Sequential(
        #     nn.Linear(input_dim * 2, 128),  # input_dim * 2 because a and v are concatenated
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        # )
        self.avattention = AVattention()

    def forward(self, a, v):
        """
        a.shape [B, N, C]
        v.shape [B, M, C]
        """
        # router_a, router_v = torch.mean(a, dim=1), torch.mean(v, dim=1)
        # # Concatenate auditory and visual modalities
        # combined_input = torch.cat((router_a, router_v), dim=-1)  # Concatenate along feature dimension
        #
        # # Router determines the expert selection based on combined input
        # gate = torch.softmax(self.router(combined_input), dim=1)
        #
        # # Select experts based on the gating scores
        # gate_a = gate[:, :1]  # Auditory experts selection
        # gate_v = gate[:, 1:]  # Visual experts selection

        # Weighted sum of expert outputs for each modality
        singlemodal_output = torch.mean(self.singlemodal_expert(a, a), dim=1)
        # multimodal_output = torch.mean(self.multimodal_expert(a, v), dim=1)
        # output = self.avattention(singlemodal_output, multimodal_output)

        return singlemodal_output


class MetamodalFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(MetamodalFusion, self).__init__()
        # self.fc_meta_a = MixtureOfExperts(input_dim=input_dim, output_dim=input_dim, num_experts=4)
        # self.fc_meta_v = MixtureOfExperts(input_dim=input_dim, output_dim=input_dim, num_experts=4)
        self.fc_out = nn.Linear(2 * input_dim, output_dim)
        self.discriminator_fc = nn.Linear(input_dim, 2)
    def forward(self, a, v, alpha):
        """
        a.shape [B, N, C]
        v.shape [B, M, C]
        """
        # output_a = self.fc_meta_a(a, v)
        # output_v = self.fc_meta_v(v, a)
        output_a, output_v = torch.mean(a, dim=1), torch.mean(v, dim=1)
        output_a_reversed = GradientReversalFunction.apply(output_a, alpha)
        output_v_reversed = GradientReversalFunction.apply(output_v, alpha)
        disc_pred_a = self.discriminator_fc(output_a_reversed)
        disc_pred_v = self.discriminator_fc(output_v_reversed)
        output = torch.cat((output_a, output_v), dim=1)
        output = self.fc_out(output)
        return output_a, output_v, disc_pred_a, disc_pred_v, output


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

