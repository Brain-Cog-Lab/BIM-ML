import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import LayerNorm
from .layers import EncoderLayer, Encoder, DecoderLayer, Decoder


class Prompt_Attention(nn.Module):
    def __init__(self, dim, prompt_dim, nhead=4, dropout=0.1):
        super().__init__()
        assert dim % nhead == 0, 'dim should be divisible by num_heads'
        self.proj1 = nn.Linear(dim + prompt_dim, dim + prompt_dim)
        self.proj2 = nn.Linear(dim + prompt_dim, dim)

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.FF_dropout = Dropout(dropout)
        self.multihead_attn = MultiheadAttention(dim, nhead, dropout=0., batch_first=True)

        # self.init_fn()


    # def init_fn(self):
    #     nn.init.kaiming_normal_(self.proj1.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.proj2.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, src, memory, prompt):
        """
            # 输入分别是query, key+value, prompt
            # 通过将prompt加到key和value上，输出模态互补的该模态特征, 与原特征(value)进行残差连接
            src: single modality query [B, T]
            memory: single modality feature [B, T]
            prompt: modality attached prompt [B, T]
        """
        specific_memory = torch.cat([prompt, memory], dim=1)
        specific_memory = self.proj2(F.relu(self.proj1(specific_memory)))

        output = self.multihead_attn(src, specific_memory, memory)[0]

        # # Add & Norm
        # output = memory + self.dropout1(res)
        # output = self.norm1(output)
        #
        ## Feed Forward
        output2 = self.linear2(self.FF_dropout(F.relu(self.linear1(output))))
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        return output


class BilinearPooling(nn.Module):
    def __init__(self):
        super(BilinearPooling, self).__init__()

    def forward(self, audio, visual):
        # audio: [B, T], visual: [B, T]
        audio = audio.unsqueeze(2)  # [B, T, 1]
        visual = visual.unsqueeze(1)  # [B, 1, T]

        bilinear = torch.bmm(audio, visual)  # [B, T, T]

        # 聚合特征：对第三维进行求和
        pooled_feature = torch.sum(bilinear, dim=2)  # [B, T]
        # 可以选择进行归一化
        pooled_feature = F.normalize(pooled_feature, p=2, dim=1)

        return pooled_feature


class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.fc_visual = nn.Linear(visual_dim, hidden_dim)
        self.fc_audio = nn.Linear(audio_dim, hidden_dim)
        self.fc_weights = nn.Linear(hidden_dim * 2, 2)  # 用于学习两个权重：视觉和听觉的注意力权重

    def forward(self, visual_input, audio_input):
        # 处理单模态输入
        visual_features = F.relu(self.fc_visual(visual_input))
        audio_features = F.relu(self.fc_audio(audio_input))

        # 拼接特征并计算注意力权重
        concatenated_features = torch.cat((visual_features, audio_features), dim=1)
        attention_weights = F.softmax(self.fc_weights(concatenated_features), dim=1)

        # 应用注意力权重
        weighted_visual = attention_weights[:, 0:1] * visual_features
        weighted_audio = attention_weights[:, 1:2] * audio_features
        combined_features = weighted_visual + weighted_audio

        return combined_features



class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)
        self.relu = nn.ReLU(inplace=True)
        self.affine_matrix = nn.Linear(d_model, d_model)

    def forward(self, query_feature, memory_feature):
        output = self.decoder(query_feature, memory_feature)
        output = self.relu(self.affine_matrix(output))
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N, C = query.shape
        B, M, C = key.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout, batch_first=True)
        # self.norm1 = nn.LayerNorm(d_model)

    def forward(self, audio_feat, video_feat, fused_feature):
        # video_feat, audio_feat: [batch, seq, 768]
        global_feat = torch.mean(video_feat, dim=1) * torch.mean(audio_feat, dim=1)  # (batch, 768)
        if fused_feature is not None:
            key_memory = torch.cat([fused_feature['key_prompt'], audio_feat, video_feat], dim=1)
            value_memory = torch.cat([fused_feature['value_prompt'], audio_feat, video_feat], dim=1)
        else:
            key_memory = torch.cat([audio_feat, video_feat], dim=1)
            value_memory = torch.cat([audio_feat, video_feat], dim=1)
        output = self.video_multihead(global_feat.unsqueeze(1), key_memory, value_memory)[0].squeeze()
        # output = self.norm1(global_feat + self.dropout(mid_out))

        return output