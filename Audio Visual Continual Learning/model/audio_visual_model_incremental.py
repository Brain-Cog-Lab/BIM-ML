import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LSCLinear, SplitLSCLinear
from .prompt import EPrompt
from .attention import Prompt_Attention, BilinearPooling, CrossModalAttention, InternalTemporalRelationModule, CrossModalRelationAttModule, Attention, AudioVideoInter


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        self.use_e_prompt = False# args.e_prompt
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)
        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)
        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)
        
        if LSC:
            self.audio_fc = nn.Linear(768, self.num_classes)
            self.visual_fc = nn.Linear(768, self.num_classes)
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.audio_fc = nn.Linear(768, self.num_classes)
            self.visual_fc = nn.Linear(768, self.num_classes)
            self.classifier = nn.Linear(768, self.num_classes)

        self.a_prompt = EPrompt(key_dim=768, prompt_dim=768)
        self.v_prompt = EPrompt(key_dim=768, prompt_dim=768)
        self.av_cue_fusion = CrossModalAttention(768, 768, 768)
        self.visual_decoder = CrossModalRelationAttModule(input_dim=768, d_model=768, feedforward_dim=768)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=768, d_model=768, feedforward_dim=768)
    
    def forward(self, visual=None, audio=None, out_logits=True, out_features=False, out_features_norm=False, out_feature_before_fusion=False, out_attn_score=False, AFC_train_out=False, is_train=True,):
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            logits = self.classifier(visual_feature)
            outputs = ()
            if AFC_train_out:
                visual_feature.retain_grad()
                outputs += (logits, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(visual_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')
            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                outputs += (logits, audio_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(audio_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs
        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            visual = visual.view(visual.shape[0], 8, -1, 768)  # [b, l, s, d] -> [256, 8, 196, 768]
            spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual)
            visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)
            visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)

            audio_feature = audio
            visual_feature = visual_pooled_feature

            if self.use_e_prompt:
                audio_feature_raw = audio_feature.unsqueeze(1).transpose(0, 1).contiguous()  # (seq, batch, dim)
                visual_feature_raw = visual_feature.unsqueeze(1).transpose(0, 1).contiguous()  # (seq, batch, dim)

                # 这里是新加的对e_prompt的操作
                prompt_mask = None  # 当前的重新计算, 之前的用之前的
                v_res = self.v_prompt(visual_feature_raw, prompt_mask=prompt_mask, cls_features=visual_feature_raw)  # 输入和查询的是一样的
                visual_feature = v_res['prompted_embedding'].transpose(0,1).contiguous()  # # (seq, batch, dim)
                a_res = self.a_prompt(audio_feature_raw, prompt_mask=prompt_mask, cls_features=audio_feature_raw)  # 输入和查询的是一样的
                audio_feature = a_res['prompted_embedding'].transpose(0,1).contiguous()  # # (seq, batch, dim)

                # visual_key_value_feature = self.visual_encoder(visual_feature)
                visual_key_value_feature = visual_feature

                audio_query_output = self.audio_decoder(audio_feature_raw, visual_key_value_feature)  # (seq, batch, dim)  表示视觉中和音频有关的视觉特征, 应该加到视觉中.

                # video query
                # audio_key_value_feature = self.audio_encoder(audio_feature)
                audio_key_value_feature = audio_feature
                visual_query_output = self.visual_decoder(visual_feature_raw, audio_key_value_feature)  # (seq, batch, dim)

                visual_feature = (visual_feature_raw + audio_query_output).squeeze()
                audio_feature = (audio_feature_raw + visual_query_output).squeeze()

            # audio_feature = F.relu(self.audio_proj(audio_feature))  # (B, C)
            audio_feature = self.audio_proj(audio_feature)  # (B, C)
            # visual_feature = F.relu(self.visual_proj(visual_feature))  # （B, C）
            visual_feature = self.visual_proj(visual_feature)  # （B, C）
            audio_visual_features = audio_feature + visual_feature

            logits = self.classifier(audio_visual_features)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_pooled_feature.retain_grad()
                outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (self.audio_fc(audio_feature), self.visual_fc(visual_feature), logits,)
                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(audio_visual_features),)
                    else:
                        outputs += (audio_visual_features,)
                if out_feature_before_fusion:
                    outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
                if out_attn_score:
                    outputs += (spatial_attn_score, temporal_attn_score)
                if is_train and self.use_e_prompt:
                    outputs += (a_res['reduce_sim'], v_res['reduce_sim'])
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

    def audio_visual_attention(self, audio_features, visual_features):

        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score
    

    def incremental_classifier(self, numclass):
        weight = self.audio_fc.weight.data
        bias = self.audio_fc.bias.data
        in_features = self.audio_fc.in_features
        out_features = self.audio_fc.out_features

        self.audio_fc = nn.Linear(in_features, numclass, bias=True)
        self.audio_fc.weight.data[:out_features] = weight
        self.audio_fc.bias.data[:out_features] = bias

        weight = self.visual_fc.weight.data
        bias = self.visual_fc.bias.data
        in_features = self.visual_fc.in_features
        out_features = self.visual_fc.out_features

        self.visual_fc = nn.Linear(in_features, numclass, bias=True)
        self.visual_fc.weight.data[:out_features] = weight
        self.visual_fc.bias.data[:out_features] = bias

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias


class IncreAudioVisualNetWithPrompt(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNetWithPrompt, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        self.use_e_prompt = args.e_prompt
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        self.audio_proj = nn.Linear(768, 768)
        self.visual_proj = nn.Linear(768, 768)

        if LSC:
            self.classifier = LSCLinear(768, self.num_classes, bias=False)
        else:
            self.classifier = nn.Linear(768, self.num_classes, bias=True)

        self.aQueryAttention = Attention(768, num_heads=1)
        self.vQueryAttention = Attention(768, num_heads=1)

        self.aKeyValuePrompt = EPrompt()
        self.vKeyValuePrompt = EPrompt()
        self.avKeyValuePrompt = EPrompt()

        self.AVInter = AudioVideoInter(768, n_head=1, head_dropout=0.0)
        self.VAInter = AudioVideoInter(768, n_head=1, head_dropout=0.0)

    def forward(self, visual=None, audio=None, out_logits=True, out_features=False, out_features_norm=False,
                out_feature_before_fusion=False, is_train=True, out_kl_features=False):
        if visual is None:
            raise ValueError('input frames are None when modality contains visual')
        if audio is None:
            raise ValueError('input audio are None when modality contains audio')

        visual = visual.view(visual.shape[0], 8, -1, 768)  # [b, l, s, d] -> [256, 8, 196, 768]
        visual = torch.mean(visual, dim=1)  # [256, 196, 768]
        audio = audio.unsqueeze(1) # [256, 1, 768]

        visual_key, visual_value = visual, visual
        audio_key, audio_value = audio, audio
        if self.use_e_prompt:
            audio_feature_raw = audio.transpose(0, 1).contiguous()  # (seq, batch, dim)
            visual_feature_raw = visual.transpose(0, 1).contiguous()  # (seq, batch, dim)

            # 这里是新加的对e_prompt的操作
            prompt_mask = None
            v_res = self.vKeyValuePrompt(visual_feature_raw, prompt_mask=prompt_mask, cls_features=visual_feature_raw)  # 输入和查询的是一样的
            # visual_key = torch.cat([v_res['key_prompt'], visual], dim=1)
            # visual_value = torch.cat([v_res['value_prompt'], visual], dim=1)

            a_res = self.aKeyValuePrompt(audio_feature_raw, prompt_mask=prompt_mask, cls_features=audio_feature_raw)  # 输入和查询的是一样的
            # audio_key = torch.cat([a_res['key_prompt'], audio], dim=1)
            # audio_value = torch.cat([a_res['value_prompt'], audio], dim=1)

        aQuery = self.aQueryAttention(audio, visual_key, visual_value)
        vQuery = self.vQueryAttention(visual, audio_key, audio_value)

        multimodal_audio_feature = aQuery + audio  # (batch, seq, dim)
        multimodal_audio_feature = torch.mean(multimodal_audio_feature, dim=1).unsqueeze(1) # (batch, 1, dim)
        audio_feature = F.relu(self.audio_proj(multimodal_audio_feature)).transpose(0, 1).contiguous().squeeze()  # (B, C)
        # audio_feature = multimodal_audio_feature.transpose(0, 1).contiguous()  # (1, B, C)
        # audio_feature = torch.mean(multimodal_audio_feature, dim=1)

        multimodal_visual_feature = vQuery #+ visual  # (batch, seq, dim)
        multimodal_visual_feature = torch.mean(multimodal_visual_feature, dim=1).unsqueeze(1)  # (batch, 1, dim)
        visual_feature = F.relu(self.visual_proj(multimodal_visual_feature)).transpose(0, 1).contiguous().squeeze()  # (B, C)
        # visual_feature = multimodal_visual_feature.transpose(0, 1).contiguous()  # （1, B, C）
        # visual_feature = torch.mean(multimodal_visual_feature, dim=1)

        fused_prompt = None
        fused_feature = audio_feature * visual_feature

        # if self.use_e_prompt:
        #     fused_prompt = self.aKeyValuePrompt(fused_feature, prompt_mask=prompt_mask, cls_features=fused_feature)

        visual_query_feature = self.VAInter(multimodal_visual_feature, multimodal_audio_feature, fused_prompt)  # (B, C)
        audio_query_feature = self.AVInter(multimodal_audio_feature, multimodal_visual_feature, fused_prompt)  # (B, C)

        # audio_feature = F.relu(self.audio_proj(audio_feature))  # (B, C)
        # visual_feature = F.relu(self.visual_proj(visual_feature))  # （B, C）
        audio_visual_features = audio_feature + visual_feature
        audio_visual_features = audio_query_feature + visual_query_feature

        logits = self.classifier(audio_visual_features)
        outputs = ()
        if out_logits:
            outputs += (logits,)
        if out_features:
            if out_features_norm:
                outputs += (F.normalize(audio_visual_features),)
            else:
                outputs += (audio_visual_features,)
        if out_feature_before_fusion:
            outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
        if out_kl_features:
            outputs += (F.normalize(torch.mean(aQuery, dim=1)), F.normalize(torch.mean(vQuery, dim=1)))
        if is_train and self.use_e_prompt:
            outputs += (a_res['reduce_sim'], v_res['reduce_sim'])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


    def incremental_classifier(self, numclass):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias
