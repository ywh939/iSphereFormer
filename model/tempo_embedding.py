import torch
import torch.nn as nn
import functools

import spconv.pytorch as spconv


class GatedUnit(nn.Module):
    def __init__(self, input_channel, middle_channel) -> None:
        super().__init__()

        self.fc = nn.Linear(input_channel, middle_channel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))

class TempoEmbedding(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_channel, output_channel)
        self.fc2 = nn.Linear(output_channel, output_channel)
        self.bn = nn.BatchNorm1d(output_channel)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn(self.fc1(x)))
        x = self.act(self.bn(self.fc2(x)))
        return x

class GatedTempoClueEncoder(nn.Module):
    def __init__(self, input_channel, middle_channel) -> None:
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, middle_channel, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(middle_channel),
            nn.ReLU(),
        )

        self.tempo_embedding = TempoEmbedding(input_channel=middle_channel, output_channel=middle_channel)
        self.gated_unit = GatedUnit(input_channel=middle_channel, middle_channel=middle_channel)

    def forward(self, cur_frame, sequence_frames):

        cur_frame = self.input_conv(cur_frame)
        all_feat = []

        for batch_data in sequence_frames:
            
            sequence_feat_list = []
            cur_spconv_feat = self.input_conv(batch_data[0]['spconv_feat'])
            sequence_feat_list.append(cur_spconv_feat.features)

            for frame in batch_data[1:]:
                frame['post_spconv_feat'] = self.input_conv(frame['spconv_feat'])
                sequence_feat_list.append(frame['post_spconv_feat'].features)
            
            sequence_feat = torch.cat(sequence_feat_list)

            tempo_feat = self.tempo_embedding(sequence_feat)

            cur_feat = cur_spconv_feat.features
            
            cur_tempo_feat = tempo_feat[:cur_feat.shape[0]]

            gated_feat = self.gated_unit(cur_feat)

            batch_data[0]['post_spconv_feat'] = cur_feat * gated_feat + (1 - gated_feat) * cur_tempo_feat

            all_feat.append(batch_data[0]['post_spconv_feat'])
        
        cur_frame.replace_feature(torch.cat(all_feat))
        return cur_frame