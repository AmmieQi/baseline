import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from modules_.bilinear import Bilinear
from modules_.dynamic_rnn import DynamicGRU

from modules_.cross_gate import CrossGate
from modules_.graph_convolution import GraphConvolution
from modules_.multihead_attention import MultiHeadAttention
from modules_.tanh_attention import TanhAttention
from utils import generate_anchors


class VideoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.video_feature_dim, args.d_model)

    def forward(self, x):
        x = x.squeeze()
        return self.linear(x)


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words
        self.rnn = DynamicGRU(args.word_dim, args.d_model >> 1, bidirectional=True, batch_first=True)

    def forward(self, x, mask):
        length = mask.sum(dim=-1)

        x = self.rnn(x, length, self.max_num_words)
        x = F.dropout(x, self.dropout, self.training)

        return x


class LNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dropout = args.dropout
        self.max_num_frames = args.max_num_frames

        self.anchors = generate_anchors(dataset=args.dataset)
        self.num_anchors = self.anchors.shape[0]
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, args.max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

        # VideoEncoder
        self.video_encoder = VideoEncoder(args)

        # SentenceEncoder
        self.sentence_encoder = SentenceEncoder(args)

        self.v2s = TanhAttention(args.d_model)
        self.cross_gate = CrossGate(args.d_model)
        # self.fc = Bilinear(args.d_model, args.d_model, args.d_model)
        self.rnn = DynamicGRU(args.d_model << 1, args.d_model >> 1, bidirectional=True, batch_first=True)

        self.fc_score = nn.Conv1d(args.d_model, self.num_anchors, kernel_size=1, padding=0, stride=1)
        self.fc_reg = nn.Conv1d(args.d_model, self.num_anchors << 1, kernel_size=1, padding=0, stride=1)

        # loss function
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.SmoothL1Loss()

    def forward(self, frames, frame_mask, words, word_mask,
                label, label_mask, gt):
        # frames: (bs, frame_len, feature_dim, 1, 1)
        # frame_mask: (bs, frame_len, )
        # words: (bs, word_len, word_embed_dim)
        # word_mask: (bs, word_len, )
        # label: (bs, proposal_len, )
        # label: (bs, proposal_len, )
        # gt: (bs, 2)
        frames_len = frame_mask.sum(dim=-1)

        frames = F.dropout(frames, self.dropout, self.training)
        words = F.dropout(words, self.dropout, self.training)

        frames = self.video_encoder(frames)  # frames: (bs, frame_len, args.d_model)

        x = self.sentence_encoder(words, word_mask)  # x: (bs, word_len, args.d_model)

        # interactive
        x1 = self.v2s(frames, x)  # x1: (bs, frame_len, args.d_model)
        frames1, x1 = self.cross_gate(frames, x1)  # frames1:(bs, frame_len, args.d_model)
        x = torch.cat([frames1, x1], -1)  # x: (bs, frame_len, d_model*2)
        # x = self.fc(frames1, x1, F.relu)
        x = self.rnn(x, frames_len, self.max_num_frames)  # x: (bs, frame_len, d_model)
        x = F.dropout(x, self.dropout, self.training)

        # loss
        predict = torch.sigmoid(self.fc_score(x.transpose(-1, -2))).transpose(-1, -2)
        # [batch, max_num_frames, num_anchors]
        reg = self.fc_reg(x.transpose(-1, -2)).transpose(-1, -2)
        reg = reg.contiguous().view(-1, self.args.max_num_frames * self.num_anchors, 2)
        # [batch, max_num_frames, num_anchors, 2]
        predict_flatten = predict.contiguous().view(predict.size(0), -1) * label_mask.float()
        cls_loss = self.criterion1(predict_flatten, label)
        # gt_box: [batch, 2]
        proposals = torch.from_numpy(self.proposals).type_as(gt).float()  # [max_num_frames, num_anchors, 2]
        proposals = proposals.view(-1, 2)
        if not self.training:
            indices = torch.argmax(predict_flatten, -1)
        else:
            indices = torch.argmax(label, -1)
        predict_box = proposals[indices]  # [nb, 2]
        predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
        refine_box = predict_box + predict_reg
        reg_loss = self.criterion2(refine_box, gt.float())
        loss = cls_loss + 1e-3 * reg_loss
        # if detail:
        #     return refine_box, loss, predict_flatten, reg, proposals
        return refine_box, loss

