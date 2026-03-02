import torch
import torch.nn as nn
import torch.nn.functional as F
import time

__all__ = ["PointLineNet", "hg"]
import numpy as np
from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent.parent / 'point_model/point_model.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        features = []
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        features.append(x)                  # [B, 64, H, W]
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        features.append(x)                  # [B, 64, H/2, W/2]
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        features.append(x)                  # [B, 128, H/4, W/4]

        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'features': features,
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


class UNet(nn.Module):

    def __init__(self, input_channel, conv_channel, output_channel, layer_num):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        D0, D1 = conv_channel, int(conv_channel/2)
        D2 = D0 + D1

        # [H/4, W/4]
        self.conv1a = nn.Conv2d(input_channel, D0, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(D0)
        self.conv1b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(D0)

        # [H/8, W/8]
        self.conv2a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(D0)
        self.conv2b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(D0)

        # [H/16, W/16]
        self.conv3a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(D0)
        self.conv3b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(D0)

        # [H/32, W/32]
        self.conv4a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(D0)
        self.conv4b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(D0)

        # [H/64, W/64]
        self.conv5a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn5a = nn.BatchNorm2d(D0)
        self.conv5b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn5b = nn.BatchNorm2d(D0)

        # [H/64, W/64]
        # self.deconv1 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn1_dec = nn.BatchNorm2d(D1)

        # [H/32, W/32]
        self.conv4a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn4a_up   = nn.BatchNorm2d(D1)
        self.conv4b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4b_up   = nn.BatchNorm2d(D0)

        # [H/32, W/32]
        # self.deconv2 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn2_dec = nn.BatchNorm2d(D1)

        # [H/16, W/16]
        self.conv3a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn3a_up   = nn.BatchNorm2d(D1)
        self.conv3b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3b_up   = nn.BatchNorm2d(D0)

        # [H/16, W/16]
        # self.deconv3 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn3_dec = nn.BatchNorm2d(D1)

        # [H/8, W/8]
        self.conv2a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn2a_up   = nn.BatchNorm2d(D1)
        self.conv2b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2b_up   = nn.BatchNorm2d(D0)

        # [H/8, W/8]
        # self.deconv4 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn4_dec = nn.BatchNorm2d(D1)

        # [H/4, W/4]
        self.conv1a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn1a_up   = nn.BatchNorm2d(D1)
        self.conv1b_up = nn.Conv2d(D0, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1b_up   = nn.BatchNorm2d(output_channel)


    def forward(self, x):
        x1 = self.relu(self.bn1a(self.conv1a(x)))          # [B, D0, H/4, W/4]
        x1 = self.relu(self.bn1b(self.conv1b(x1)))          # [B, D0, H/4, W/4]

        x2 = self.pool(x1)                                 # [B, D0, H/8, W/8]

        x2 = self.relu(self.bn2a(self.conv2a(x2)))          # [B, D0, H/8, W/8]
        x2 = self.relu(self.bn2b(self.conv2b(x2)))          # [B, D0, H/8, W/8]

        x3 = self.pool(x2)                                 # [B, D0, H/16, W/16]

        x3 = self.relu(self.bn3a(self.conv3a(x3)))          # [B, D0, H/16, W/16]
        x3 = self.relu(self.bn3b(self.conv3b(x3)))          # [B, D0, H/16, W/16]

        x4 = self.pool(x3)                                 # [B, D0, H/32, W/32]

        x4 = self.relu(self.bn4a(self.conv4a(x4)))          # [B, D0, H/32, W/32]
        x4 = self.relu(self.bn4b(self.conv4b(x4)))          # [B, D0, H/32, W/32]

        x5 = self.pool(x4)                                 # [B, D0, H/64, W/64]

        x5 = self.relu(self.bn5a(self.conv5a(x5)))          # [B, D0, H/64, W/64]
        x5 = self.relu(self.bn5b(self.conv5b(x5)))          # [B, D0, H/64, W/64]

        x = F.interpolate(x5, scale_factor=2)
        x = self.relu(self.bn1_dec(self.deconv1(x)))           # [B, D1, H/32, W/32]

        x4_up = self.relu(self.bn4a_up(self.conv4a_up(x4)))     # [B, D1, H/32, W/32]
        x = torch.cat([x, x4_up], -3)                           # [B, D0, H/32, W/32]
        x = self.relu(self.bn4b_up(self.conv4b_up(x)))          # [B, D0, H/32, W/32]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn2_dec(self.deconv2(x)))            # [B, D1, H/16, W/16]

        x3_up = self.relu(self.bn3a_up(self.conv3a_up(x3)))     # [B, D1, H/16, W/16]
        x = torch.cat([x, x3_up], -3)                           # [B, D0, H/16, W/16]
        x = self.relu(self.bn3b_up(self.conv3b_up(x)))          # [B, D0, H/16, W/16]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn3_dec(self.deconv3(x)))            # [B, D1, H/8, W/8]

        x2_up = self.relu(self.bn2a_up(self.conv2a_up(x2)))     # [B, D1, H/8, W/8]
        x = torch.cat([x, x2_up], -3)                           # [B, D0, H/8, W/8]
        x = self.relu(self.bn2b_up(self.conv2b_up(x)))          # [B, D0, H/8, W/8]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn4_dec(self.deconv4(x)))            # [B, D1, H/4, W/4]

        x1_up = self.relu(self.bn1a_up(self.conv1a_up(x1)))     # [B, D1, H/4, W/4]
        x = torch.cat([x, x1_up], -3)                           # [B, D0, H/4, W/4]
        x = self.relu(self.bn1b_up(self.conv1b_up(x)))          # [B, output_channel, H/4, W/4]

        return x
    
#点-线跨任务增强
class AdaptiveChannelPrompt(nn.Module):
    """
    Drop-in replacement for the original AdaptiveChannelPrompt.

    Inputs:
      - backbone_feat: [B, C, H, W], C == feature_channels
      - stage1_pred:   [B, 9, H, W] (first 5: line, last 4: junction)

    Outputs (unchanged):
      - junction_enhanced_feat: [B, C, H, W]  # line -> junction 引导后的特征
      - line_enhanced_feat:     [B, C, H, W]  # junction -> line   引导后的特征
    """

    def __init__(self, feature_channels=256, reduction=16, detach_stage1=False, alpha_init=0.0):
        super().__init__()
        C = feature_channels
        r = max(1, reduction)
        hidden = max(16, C // r)  # 避免过窄瓶颈

        # 与原实现保持一致的类别拆分
        self.line_cls = 5
        self.junc_cls = 4

        # 是否切断对 stage1 的反传（默认 False，行为与原实现一致）
        self.detach_stage1 = detach_stage1

        # -------- 线 -> 点：（用 line_pred 引导 point/junction 分支）--------
        # 通道门控（GAP -> 1x1 -> ReLU -> 1x1 -> Sigmoid）
        self.l2j_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.line_cls, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C, 1, bias=True),
            nn.Sigmoid()
        )
        # 空间门控（1x1 -> Sigmoid）
        self.l2j_s = nn.Sequential(
            nn.Conv2d(self.line_cls, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # -------- 点 -> 线：（用 junc_pred 引导 line 分支）--------
        self.j2l_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.junc_cls, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C, 1, bias=True),
            nn.Sigmoid()
        )
        self.j2l_s = nn.Sequential(
            nn.Conv2d(self.junc_cls, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 可学习的门控强度；默认 0 → 初期为恒等映射，更稳
        self.alpha_c = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.alpha_s = nn.Parameter(torch.full((1,), float(alpha_init)))


    def forward(self, backbone_feat, stage1_pred):
        # stage1_pred: [B, 9, H, W]  (前5线, 后4点)
        line_pred = stage1_pred[:, :self.line_cls]
        junc_pred = stage1_pred[:, self.line_cls:]

        if self.detach_stage1:
            line_pred = line_pred.detach()
            junc_pred = junc_pred.detach()

        # ------ 线 -> 点 的跨任务注意力 ------
        wc_j = self.l2j_c(line_pred)   # [B, C, 1, 1], in (0,1)
        ws_j = self.l2j_s(line_pred)   # [B, 1, H, W], in (0,1)

        # ------ 点 -> 线 的跨任务注意力 ------
        wc_l = self.j2l_c(junc_pred)   # [B, C, 1, 1], in (0,1)
        ws_l = self.j2l_s(junc_pred)   # [B, 1, H, W], in (0,1)

        # 允许“放大/抑制”：把 [0,1] 映射到 [-1,1]
        wc_j = 2.0 * wc_j - 1.0
        ws_j = 2.0 * ws_j - 1.0
        wc_l = 2.0 * wc_l - 1.0
        ws_l = 2.0 * ws_l - 1.0

        # 以“1”为基线的可加性缩放（恒等起步 → 学到再偏移）
        # 先通道门控，再空间门控；两者共享强度系数，保持简洁稳定
        junc_enhanced_feat = backbone_feat * (1.0 + self.alpha_c * wc_j)
        junc_enhanced_feat = junc_enhanced_feat * (1.0 + self.alpha_s * ws_j)

        line_enhanced_feat = backbone_feat * (1.0 + self.alpha_c * wc_l)
        line_enhanced_feat = line_enhanced_feat * (1.0 + self.alpha_s * ws_l)

        # 返回顺序与原实现完全一致
        return junc_enhanced_feat, line_enhanced_feat

#结合几何信息的融合 独立的注意力机制实现：执行最终、最深层的特征融合
class LightweightCrossAttention(nn.Module):
    """
    稀疏轻量级Cross-Attention模块 - 实现点线特征与原始特征的交互融合
    设计目标：latency ~5ms，通过稀疏注意力和降维实现
    """
    def __init__(self, feature_channels=256, num_heads=4, reduction_ratio=8, 
                 sparse_ratio=0.1, window_size=8):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.sparse_ratio = sparse_ratio  # 稀疏注意力比例
        self.window_size = window_size    # 局部窗口大小
        
        # 降维 256 -> 32
        self.reduced_channels = feature_channels // reduction_ratio  # 256 -> 32
        self.head_dim = self.reduced_channels // num_heads  # 32 -> 8 per head
        
        assert self.reduced_channels % num_heads == 0, "reduced_channels must be divisible by num_heads"
        
        # 输入投影：将原始特征降维
        self.input_proj = nn.Conv2d(feature_channels, self.reduced_channels, 1, bias=False)
        
        # Query/Key/Value投影
        self.q_proj = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1, bias=False)
        
        # 输出投影：恢复到原始维度
        self.output_proj = nn.Conv2d(self.reduced_channels, feature_channels, 1, bias=False)
        
        # Layer Norm（在降维空间操作，减少计算）
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=self.reduced_channels)  # 32通道用4组
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=self.reduced_channels)
        
        # 轻量级FFN（进一步减少计算）
        self.ffn = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 1),  # 不再扩展
            nn.ReLU(inplace=True)
        )
        
        # 门控机制（控制融合强度）
        # 输入是backbone_feat和point/line_feat的连接，所以是2*feature_channels
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels // 8, 1),  # 进一步压缩
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """保守初始化：确保网络从恒等映射开始学习"""
        for m in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        
        # 输出投影初始化为接近0，确保残差连接稳定
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        
        # FFN保守初始化 - 现在只有2层
        nn.init.xavier_uniform_(self.ffn[0].weight, gain=0.1)
        # ffn[1]是ReLU，没有权重需要初始化
    
    def sparse_cross_attention(self, query_feat, key_feat, value_feat):
        """
        稀疏Cross-Attention计算 - 使用局部窗口+top-k选择
        Args:
            query_feat: [B, C_reduced, H, W] Query特征
            key_feat: [B, C_reduced, H, W] Key特征  
            value_feat: [B, C_reduced, H, W] Value特征
        Returns:
            output: [B, C_reduced, H, W] 注意力输出
        """
        B, C, H, W = query_feat.shape
        
        # 投影Q,K,V
        Q = self.q_proj(query_feat)  # [B, C_reduced, H, W]
        K = self.k_proj(key_feat)    # [B, C_reduced, H, W]
        V = self.v_proj(value_feat)  # [B, C_reduced, H, W]
        
        # === 方法1：局部窗口注意力（类似Swin Transformer） ===
        if self.window_size > 0:
            return self._window_attention(Q, K, V, H, W)
        
        # === 方法2：Top-k稀疏注意力 ===
        else:
            return self._topk_attention(Q, K, V, H, W)
    
    def _window_attention(self, Q, K, V, H, W):
        """局部窗口注意力 - 优化内存使用"""
        B, C = Q.shape[:2]
        ws = self.window_size
        
        # 确保H,W能被窗口大小整除（简单处理：截断而不是padding）
        H_win = (H // ws) * ws
        W_win = (W // ws) * ws
        
        if H_win != H or W_win != W:
            Q = Q[:, :, :H_win, :W_win]
            K = K[:, :, :H_win, :W_win]
            V = V[:, :, :H_win, :W_win]
        
        # 重新组织为窗口
        # [B, C, H, W] -> [B, num_heads, head_dim, H_win, W_win]
        Q = Q.view(B, self.num_heads, self.head_dim, H_win, W_win)
        K = K.view(B, self.num_heads, self.head_dim, H_win, W_win)
        V = V.view(B, self.num_heads, self.head_dim, H_win, W_win)
        
        # 分割成窗口 [B, num_heads, head_dim, num_h_win, ws, num_w_win, ws]
        Q = Q.view(B, self.num_heads, self.head_dim, H_win//ws, ws, W_win//ws, ws)
        K = K.view(B, self.num_heads, self.head_dim, H_win//ws, ws, W_win//ws, ws)
        V = V.view(B, self.num_heads, self.head_dim, H_win//ws, ws, W_win//ws, ws)
        
        # 重排列为 [B, num_heads, num_h_win, num_w_win, head_dim, ws, ws]
        Q = Q.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        K = K.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        V = V.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        
        # 重塑为窗口内的序列 [B*num_heads*num_windows, head_dim, ws*ws]
        num_windows = (H_win//ws) * (W_win//ws)
        Q = Q.view(B * self.num_heads * num_windows, self.head_dim, ws * ws)
        K = K.view(B * self.num_heads * num_windows, self.head_dim, ws * ws)
        V = V.view(B * self.num_heads * num_windows, self.head_dim, ws * ws)
        
        # 注意力计算 [B*num_heads*num_windows, ws*ws, head_dim]
        Q = Q.transpose(-2, -1)  # [B*num_heads*num_windows, ws*ws, head_dim]
        V = V.transpose(-2, -1)  # [B*num_heads*num_windows, ws*ws, head_dim]
        
        # 计算attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K) * scale  # [B*num_heads*num_windows, ws*ws, ws*ws]
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)  # [B*num_heads*num_windows, ws*ws, head_dim]
        
        # 重塑回原始格式
        attended = attended.transpose(-2, -1)  # [B*num_heads*num_windows, head_dim, ws*ws]
        attended = attended.view(B, self.num_heads, H_win//ws, W_win//ws, self.head_dim, ws, ws)
        attended = attended.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # [B, num_heads, head_dim, H_win//ws, ws, W_win//ws, ws]
        attended = attended.view(B, C, H_win, W_win)
        
        # 如果有截断，需要pad回原始大小
        if H_win != H or W_win != W:
            pad_h = H - H_win
            pad_w = W - W_win
            attended = torch.nn.functional.pad(attended, (0, pad_w, 0, pad_h))
        
        return attended
    
    def _topk_attention(self, Q, K, V, H, W):
        """Top-k稀疏注意力"""
        B, C = Q.shape[:2]
        
        # Reshape为multi-head格式
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)  # [B, heads, HW, head_dim]
        K = K.view(B, self.num_heads, self.head_dim, H * W)                     # [B, heads, head_dim, HW]
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)  # [B, heads, HW, head_dim]
        
        # 计算attention scores
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K) * scale  # [B, heads, HW, HW]
        
        # Top-k稀疏化：只保留每行的top-k个最大值
        k = max(1, int(H * W * self.sparse_ratio))  # 稀疏比例
        topk_values, topk_indices = torch.topk(scores, k, dim=-1)  # [B, heads, HW, k]
        
        # 创建稀疏mask
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        
        # Softmax + 应用attention
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)  # [B, heads, HW, head_dim]
        
        # Reshape回原始格式
        attended = attended.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return attended
    
    def forward(self, backbone_feat, point_feat, line_feat):
        """
        前向传播：实现点线特征与原始特征的交互融合
        Args:
            backbone_feat: [B, 256, H, W] 原始骨干特征
            point_feat: [B, 256, H, W] 点增强特征
            line_feat: [B, 256, H, W] 线增强特征
        Returns:
            fused_point_feat: [B, 256, H, W] 融合后的点特征
            fused_line_feat: [B, 256, H, W] 融合后的线特征
        """
        # 降维（减少计算量）
        backbone_reduced = self.input_proj(backbone_feat)  # [B, 32, H, W]
        point_reduced = self.input_proj(point_feat)        # [B, 32, H, W]
        line_reduced = self.input_proj(line_feat)          # [B, 32, H, W]
        
        # === 点特征与原始特征的稀疏Cross-Attention ===
        point_attn1 = self.sparse_cross_attention(point_reduced, backbone_reduced, backbone_reduced)
        point_attn1 = self.norm1(point_attn1 + point_reduced)  # 残差连接
        
        # 轻量级FFN
        point_ffn = self.ffn(point_attn1)
        point_enhanced = self.norm2(point_ffn + point_attn1)  # 残差连接
        
        # === 线特征与原始特征的稀疏Cross-Attention ===
        line_attn1 = self.sparse_cross_attention(line_reduced, backbone_reduced, backbone_reduced)
        line_attn1 = self.norm1(line_attn1 + line_reduced)  # 残差连接
        
        # 轻量级FFN
        line_ffn = self.ffn(line_attn1)
        line_enhanced = self.norm2(line_ffn + line_attn1)  # 残差连接
        
        # 恢复到原始维度
        point_output = self.output_proj(point_enhanced)  # [B, 256, H, W]
        line_output = self.output_proj(line_enhanced)    # [B, 256, H, W]
        
        # 门控融合（控制attention影响的强度）
        point_gate = self.gate(torch.cat([backbone_feat, point_feat], dim=1))
        line_gate = self.gate(torch.cat([backbone_feat, line_feat], dim=1))
        
        # 最终输出：原始特征 + 门控的attention特征
        fused_point_feat = backbone_feat + point_gate * point_output
        fused_line_feat = backbone_feat + line_gate * line_output
        
        return fused_point_feat, fused_line_feat

#低延迟点线交互
class LightweightCoordinatePromptFusion(nn.Module):
    def __init__(self, feature_channels=256, top_ratio=0.1, max_line_candidates=200, coord_channels=16):
        super().__init__()
        self.feature_channels = feature_channels
        self.top_ratio = top_ratio
        self.max_line_candidates = max_line_candidates
        self.use_residual = 0
        self.min_line_length = 3.0    # 最小线段长度（像素）
        self.max_line_length = 40.0   # 最大线段长度（避免过长的噪声线段）
        coord_dim = coord_channels  # 16
        
        # 坐标编码器：
        self.line_coord_encoder = nn.Sequential(
            nn.Conv2d(4, coord_dim, 3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_dim, coord_dim, 1)              
        )  
        
        self.point_coord_encoder = nn.Sequential(
            nn.Conv2d(2, coord_dim, 3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_dim, coord_dim, 1)              
        )  
        
        # 使用深度可分离卷积减少参数
        # 输入：256 + 16 = 272通道
        input_dim = feature_channels + coord_dim  # 272
        
        self.junction_fusion = nn.Sequential(
            nn.Conv2d(input_dim, 64, 1),                   
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),     
            nn.Conv2d(64, feature_channels, 1),            
        )  
        
        self.line_fusion = nn.Sequential(
            nn.Conv2d(input_dim, 64, 1),                    
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, padding=1, groups=64),     
            nn.Conv2d(64, feature_channels, 1),             
        )  
        
        # 添加轻量级Cross-Attention模块 - 使用最佳性能配置
        self.cross_attention = LightweightCrossAttention(
            feature_channels=feature_channels,
            num_heads=4,
            reduction_ratio=8,       # 256->32通道，平衡性能和精度  
            sparse_ratio=0.1,        # 稀疏比例（用于top-k模式，实际使用窗口模式）
            window_size=8            # 8x8局部窗口，测试最佳配置
        )  


    def _extract_debug_info(self, lines_coords, junctions_coords, junctions_confidence, stage1_pred):
        """
        提取用于可视化的调试信息 - 保持特征图坐标系
        """
        batch_size, _, height, width = lines_coords.shape
        debug_info = []
        
        for b in range(batch_size):
            # 提取有效线段 (非零坐标，保持特征图坐标系)
            lines_b = lines_coords[b].detach().permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
            valid_lines_mask = (lines_b.sum(dim=1) != 0)  # 过滤全零行
            valid_lines = lines_b[valid_lines_mask]  # [N_lines, 4] - 特征图坐标
            
            # 提取有效关键点 (基于置信度，保持特征图坐标系)
            confidence_threshold = 0.1
            conf_map = junctions_confidence[b, 0].detach()  # [H, W]
            high_conf_mask = conf_map > confidence_threshold  # [H, W]
            y_indices, x_indices = torch.where(high_conf_mask)
            
            if len(y_indices) > 0:
                # 获取关键点坐标 (特征图坐标系)
                points_x = junctions_coords[b, 0, y_indices, x_indices].detach()  # x坐标
                points_y = junctions_coords[b, 1, y_indices, x_indices].detach()  # y坐标
                valid_points = torch.stack([points_x, points_y], dim=1)  # [N_points, 2]
                
                # 获取对应的置信度
                point_confidences = conf_map[y_indices, x_indices]
            else:
                valid_points = torch.zeros((0, 2), device=lines_coords.device)
                point_confidences = torch.zeros((0,), device=lines_coords.device)
            
            debug_info.append({
                'lines': valid_lines.cpu(),                    # [N_lines, 4] - 特征图坐标
                'points': valid_points.cpu(),                  # [N_points, 2] - 特征图坐标
                'point_confidences': point_confidences.cpu(), # [N_points] - 置信度
                'feature_size': (height, width),               # 特征图尺寸
            })
        
        return debug_info
    

    def full_hafm_decoding(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """
        完整的HAFM解码 - 不做任何筛选，保持原始解码逻辑
        
        Args:
            md_maps: [B, 3, H, W] 角度预测 [θ, θ1, θ2]
            dis_maps: [B, 1, H, W] 距离场预测
            residual_maps: [B, 1, H, W] 残差预测
            scale: HAFM解码的scale参数
        
        Returns:
            lines_coords: [B, 4, H, W] 完整的线段坐标解码
        """
        device = md_maps.device
        batch_size, _, height, width = md_maps.shape
        
        # 网格坐标
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()
        y0, x0 = torch.meshgrid(_y, _x, indexing='ij')
        y0 = y0[None, None]  # [1, 1, H, W]
        x0 = x0[None, None]  # [1, 1, H, W]
        
        # 处理残差
        if residual_maps is not None and self.use_residual:
            sign_pad = torch.arange(-self.use_residual, self.use_residual + 1, 
                                device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
            residual = residual_maps * sign_pad
            distance_fields = dis_maps + residual
        else:
            distance_fields = dis_maps
        
        distance_fields = distance_fields.clamp(min=0, max=1.0)
        
        # 角度解码（保持原始逻辑）
        md_un = (md_maps[:, :1] - 0.5) * np.pi * 2    # θ ∈ [-π, π]
        st_un = md_maps[:, 1:2] * np.pi / 2.0         # θ1 ∈ [0, π/2]
        ed_un = -md_maps[:, 2:3] * np.pi / 2.0        # θ2 ∈ [0, -π/2]
        
        # 三角函数计算
        cs_md = md_un.cos()
        ss_md = md_un.sin()
        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)
        
        # 端点计算
        x_st_rotated = (cs_md - ss_md * y_st) * distance_fields * scale
        y_st_rotated = (ss_md + cs_md * y_st) * distance_fields * scale
        x_ed_rotated = (cs_md - ss_md * y_ed) * distance_fields * scale
        y_ed_rotated = (ss_md + cs_md * y_ed) * distance_fields * scale
        
        # 转换到全局坐标
        x_st_final = (x_st_rotated + x0).clamp(min=0, max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0, max=height-1)
        x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height-1)
        
        # 组合线段坐标
        lines_full = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final), dim=-1)
        
        # 处理residual情况
        if residual_maps is not None and self.use_residual:
            lines_full = lines_full.mean(dim=1)  # [B, H, W, 4]
        else:
            lines_full = lines_full.squeeze(1)   # [B, H, W, 4]
        
        lines_full = lines_full.permute(0, 3, 1, 2)  # [B, 4, H, W]
        return lines_full
    def simple_junction_decode(self, stage1_pred):
        """
        修正版本的关键点坐标解码，与models.py保持一致
    
        Args:
            stage1_pred: [B, 9, H, W] Stage1预测结果
    
        Returns:
            junctions_coords: [B, 2, H, W] 关键点坐标 (特征图坐标系)
            junctions_confidence: [B, 1, H, W] 关键点置信度
        """
        # 1. 正确提取关键点信息（与models.py保持一致）
        jloc_logits = stage1_pred[:, 5:7]  # [B, 2, H, W] - 背景+关键点logits
        jloc_pred = jloc_logits.softmax(1)[:, 1:]  # [B, 1, H, W] - 关键点概率（去掉背景）
        
        joff_raw = stage1_pred[:, 7:9]  # [B, 2, H, W] - 原始偏移预测
        joff_pred = joff_raw.sigmoid() - 0.5  # [B, 2, H, W] - 偏移量 [-0.5, 0.5]
        
        B, _, H, W = stage1_pred.shape
        device = stage1_pred.device
        
        # 2. 生成基础网格坐标
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 3. 基础坐标 + 亚像素偏移
        base_x = x_grid[None, None].float()  # [1, 1, H, W]
        base_y = y_grid[None, None].float()  # [1, 1, H, W]
        
        refined_x = base_x + joff_pred[:, 0:1]  # [B, 1, H, W] - x坐标 + x偏移
        refined_y = base_y + joff_pred[:, 1:2]  # [B, 1, H, W] - y坐标 + y偏移
        
        junctions_coords = torch.cat([refined_x, refined_y], dim=1)  # [B, 2, H, W]
        
        return junctions_coords, jloc_pred
        
    def junction_guided_line_filtering_with_length_and_gradient_detach(self, full_lines, junctions_coords, junctions_confidence):
        """
        统一的线段筛选：长度过滤 + 关键点引导 + 梯度分离
    
        Args:
            full_lines: [B, 4, H, W] 完整HAFM解码的线段
            junctions_coords: [B, 2, H, W] 关键点坐标 
            junctions_confidence: [B, 1, H, W] 关键点置信度
            
        Returns:
            filtered_lines: [B, 4, H, W] 经过筛选的高质量线段
        """
        device = full_lines.device
        batch_size, _, height, width = full_lines.shape
        
        # 筛选参数
        junction_radius = 2.0
        confidence_threshold = 0.1
        
        # 【关键优化】梯度分离：detach坐标用于距离计算，不传播梯度，节省显存
        lines_for_filtering = full_lines.detach()
        junctions_for_filtering = junctions_coords.detach()
        confidence_for_filtering = junctions_confidence.detach()
        
        # 1. 预处理：提取高置信度关键点位置
        high_conf_mask = confidence_for_filtering[:, 0] > confidence_threshold  # [B, H, W]
        
        # 2. 重塑线段数据以便批量处理
        lines_flat = lines_for_filtering.permute(0, 2, 3, 1).reshape(batch_size, height * width, 4)  # [B, H*W, 4]
        
        # 3. 【第一阶段筛选】长度过滤：计算线段长度并过滤
        lengths_flat = torch.sqrt(
            (lines_flat[:, :, 2] - lines_flat[:, :, 0])**2 + 
            (lines_flat[:, :, 3] - lines_flat[:, :, 1])**2
        )  # [B, H*W]
    
        # 综合mask：非零线段 + 合理长度
        non_zero_mask = (lines_flat.sum(dim=-1) != 0)  # [B, H*W]
        valid_length_mask = (lengths_flat >= self.min_line_length) & (lengths_flat <= self.max_line_length)
        valid_mask = non_zero_mask & valid_length_mask  # [B, H*W]
        
        # 4. 创建结果tensor（基于原始full_lines，保持梯度）
        filtered_lines = torch.zeros_like(full_lines)
        
        for b in range(batch_size):
            if not high_conf_mask[b].any() or not valid_mask[b].any():
                continue
                
            # 获取关键点坐标（detached版本用于计算）
            junc_y, junc_x = torch.where(high_conf_mask[b])
            if len(junc_y) == 0:
                continue
                
            junction_coords = torch.stack([
                junctions_for_filtering[b, 0, junc_y, junc_x],  # x坐标
                junctions_for_filtering[b, 1, junc_y, junc_x]   # y坐标
            ], dim=1)  # [N_junctions, 2]
            
            # 获取通过长度筛选的有效线段（detached版本用于计算）
            valid_lines = lines_flat[b][valid_mask[b]]  # [N_valid, 4]
            valid_positions = torch.where(valid_mask[b])[0]  # [N_valid]
            
            if len(valid_lines) == 0:
                continue
            
            print(f"Batch {b}: After length filtering: {len(valid_lines)} lines, {len(junction_coords)} junctions")
            
            # 5. 【第二阶段筛选】关键点引导过滤：两个端点都必须在关键点附近
            endpoints1 = valid_lines[:, :2]  # [N_valid, 2] - (x1, y1)
            endpoints2 = valid_lines[:, 2:]  # [N_valid, 2] - (x2, y2)
            
            # 【核心】计算距离（不传播梯度）
            dist_matrix1 = torch.cdist(endpoints1, junction_coords)  # [N_valid, N_junctions]
            dist_matrix2 = torch.cdist(endpoints2, junction_coords)  # [N_valid, N_junctions]
            
            # 检查两端点是否都靠近关键点
            min_dist1, _ = dist_matrix1.min(dim=1)  # [N_valid]
            min_dist2, _ = dist_matrix2.min(dim=1)  # [N_valid]
            
            # 双端点条件：两个端点都必须在关键点附近
            valid_condition = (min_dist1 <= junction_radius) & (min_dist2 <= junction_radius)
            qualified_indices = torch.where(valid_condition)[0]
            
            print(f"Batch {b}: After junction filtering: {len(qualified_indices)} lines")
            
            # 6. 【第三阶段筛选】数量限制：基于质量分数排序
            if len(qualified_indices) > self.max_line_candidates:
                # 使用综合距离作为质量分数
                quality_scores = 1.0 / (min_dist1[qualified_indices] + min_dist2[qualified_indices] + 1e-6)
                _, top_indices = torch.topk(quality_scores, self.max_line_candidates)
                qualified_indices = qualified_indices[top_indices]
                
                print(f"Batch {b}: After quantity limit: {len(qualified_indices)} lines")
            
            # 7. 将选中位置的原始线段（有梯度）复制到结果中
            if len(qualified_indices) > 0:
                original_positions = valid_positions[qualified_indices]
                pos_h = original_positions // width
                pos_w = original_positions % width
                
                # 【关键优化】使用原始full_lines（有梯度）
                selected_lines = full_lines[b, :, pos_h, pos_w]  # [4, N_selected]
                filtered_lines[b, :, pos_h, pos_w] = selected_lines
    
        return filtered_lines
    
    def external_quality_guided_line_filtering(self, full_lines, junctions_coords, junctions_confidence, 
                                            backbone_features, external_modules):
        """
        使用外部传入的模块进行完整的质量评估和筛选
        """
        device = full_lines.device
        batch_size, _, height, width = full_lines.shape
        
        # 梯度分离优化
        lines_for_filtering = full_lines.detach()
        junctions_for_filtering = junctions_coords.detach()
        confidence_for_filtering = junctions_confidence.detach()
        
        # 基础筛选参数
        junction_radius = 2.0
        confidence_threshold = 0.1
        
        filtered_lines = torch.zeros_like(full_lines)
        
        # 【新增】调试统计信息
        debug_stats = {
            'total_before_filtering': 0,
            'after_length_filtering': 0,
            'after_junction_filtering': 0,
            'after_quality_filtering': 0,
            'final_selected': 0
        }
        
        # 检查是否传入了所有需要的模块
        required_keys = ['fc1', 'fc3', 'fc4', 'fc2', 'fc2_head', 'fc2_res', 
                        'wireframe_matcher', 'bilinear_sampling', 'compute_loi_features', 'loi_cls_type']
        if external_modules is None or not all(key in external_modules for key in required_keys):
            print("⚠️  External modules not available, falling back to geometric filtering")
            return self.junction_guided_line_filtering_with_length_and_gradient_detach(
                full_lines, junctions_coords, junctions_confidence
            )
        
        print("🔍 Starting external quality-guided line filtering...")
        
        for b in range(batch_size):
            print(f"\n--- Batch {b} ---")
            
            # 1. 提取高置信度关键点
            high_conf_mask = confidence_for_filtering[b, 0] > confidence_threshold
            if not high_conf_mask.any():
                print(f"❌ No high-confidence junctions found (threshold: {confidence_threshold})")
                continue
                
            junc_y, junc_x = torch.where(high_conf_mask)
            junction_coords = torch.stack([
                junctions_for_filtering[b, 0, junc_y, junc_x],
                junctions_for_filtering[b, 1, junc_y, junc_x]
            ], dim=1)
            
            print(f"📍 Found {len(junction_coords)} high-confidence junctions")
            
            # 2. 基础长度和几何筛选
            lines_flat = lines_for_filtering[b].permute(1, 2, 0).reshape(-1, 4)
            
            # 统计原始线段数量
            non_zero_mask = (lines_flat.sum(dim=1) != 0)
            total_lines = non_zero_mask.sum().item()
            debug_stats['total_before_filtering'] += total_lines
            print(f"📊 Total decoded lines: {total_lines}")
            
            # 长度筛选
            lengths = torch.sqrt((lines_flat[:, 2] - lines_flat[:, 0])**2 + 
                            (lines_flat[:, 3] - lines_flat[:, 1])**2)
            length_valid = (lengths >= self.min_line_length) & (lengths <= self.max_line_length)
            valid_mask = non_zero_mask & length_valid
            
            valid_lines = lines_flat[valid_mask]
            valid_positions = torch.where(valid_mask)[0]
            
            after_length = len(valid_lines)
            debug_stats['after_length_filtering'] += after_length
            print(f"📏 After length filtering [{self.min_line_length}-{self.max_line_length}px]: {after_length} lines")
            
            if len(valid_lines) == 0:
                print("❌ No lines passed length filtering")
                continue
            
            # 关键点引导筛选
            endpoints1 = valid_lines[:, :2]
            endpoints2 = valid_lines[:, 2:]
            
            dist_matrix1 = torch.cdist(endpoints1, junction_coords)
            dist_matrix2 = torch.cdist(endpoints2, junction_coords)
            
            min_dist1, _ = dist_matrix1.min(dim=1)
            min_dist2, _ = dist_matrix2.min(dim=1)
            
            junction_valid = (min_dist1 <= junction_radius) & (min_dist2 <= junction_radius)
            junction_filtered_lines = valid_lines[junction_valid]
            junction_filtered_positions = valid_positions[junction_valid]
            
            after_junction = len(junction_filtered_lines)
            debug_stats['after_junction_filtering'] += after_junction
            print(f"🎯 After junction filtering (radius: {junction_radius}px): {after_junction} lines")
            
            if len(junction_filtered_lines) == 0:
                print("❌ No lines passed junction filtering")
                continue
            
            # 3. 【关键】使用外部模块进行完整的质量评估流程
            try:
                print("🧠 Running quality assessment...")
                
                # 计算LOI特征 (类似models.py中的流程)
                loi_features = external_modules['fc1'](backbone_features[b:b+1])      # [1, dim_junction, H, W]
                loi_features_thin = external_modules['fc3'](backbone_features[b:b+1])  # [1, dim_edge, H, W]
                loi_features_aux = external_modules['fc4'](backbone_features[b:b+1])   # [1, dim_edge, H, W]
                
                # 转换线段格式：从筛选后的线段中构造juncs_pred和lines_pred
                lines_pred = junction_filtered_lines.detach()  # [N, 4]
                
                # 从线段端点构造关键点
                endpoints = torch.cat([lines_pred[:, :2], lines_pred[:, 2:]], dim=0)  # [2N, 2]
                juncs_pred, unique_indices = torch.unique(endpoints, dim=0, return_inverse=True)
                
 
                lines_adjusted = lines_pred  # 与models.py不同这里：这里我们不做复杂匹配，把线段端点挪到关键点上，保留二者的差距
                lines_init = lines_pred
                
                print(f"🔧 Computing features for {len(lines_adjusted)} lines...")
                
                # 计算线段特征
                e1_features = external_modules['bilinear_sampling'](
                    loi_features[0], lines_adjusted[:, :2] - 0.5
                ).t()
                e2_features = external_modules['bilinear_sampling'](
                    loi_features[0], lines_adjusted[:, 2:] - 0.5  
                ).t()
                
                f1 = external_modules['compute_loi_features'](
                    loi_features_thin[0], lines_adjusted
                )
                f2 = external_modules['compute_loi_features'](
                    loi_features_aux[0], lines_init
                )
                
                # 特征融合和分类
                line_features = torch.cat((e1_features, e2_features, f1, f2), dim=-1)
                logits = external_modules['fc2_head'](
                    external_modules['fc2'](line_features) + 
                    external_modules['fc2_res'](torch.cat((f1, f2), dim=-1))
                )
                
                # 计算分数
                if external_modules['loi_cls_type'] == 'softmax':
                    scores = logits.softmax(dim=-1)[:, 1]
                else:
                    scores = logits.sigmoid()[:, 0]
                
                # 显示质量分数统计
                scores_mean = scores.mean().item()
                scores_std = scores.std().item()
                scores_max = scores.max().item()
                scores_min = scores.min().item()
                print(f"📈 Quality scores - Mean: {scores_mean:.3f}, Std: {scores_std:.3f}, Range: [{scores_min:.3f}, {scores_max:.3f}]")
                
                after_quality = len(scores)
                debug_stats['after_quality_filtering'] += after_quality
                
                # 按分数排序并选择Top-K
                sarg = torch.argsort(scores, descending=True)
                num_keep = min(len(sarg), self.max_line_candidates)
                top_indices = sarg[:num_keep]
                
                final_lines = lines_adjusted[top_indices]
                final_positions = junction_filtered_positions[top_indices]
                
                # 显示选中的Top-K分数
                top_scores = scores[top_indices]
                if len(top_scores) > 0:
                    print(f"🏆 Selected top {len(top_scores)} lines, scores range: [{top_scores.min():.3f}, {top_scores.max():.3f}]")
                
                final_count = len(final_lines)
                debug_stats['final_selected'] += final_count
                print(f"✅ Quality assessment successful: {final_count} high-quality lines selected")
                
            except Exception as e:
                print(f"❌ External quality detection failed: {e}")
                print("🔄 Falling back to geometric filtering...")
                
                # 出错时回退到几何筛选
                geometric_scores = 1.0 / (min_dist1[junction_valid] + min_dist2[junction_valid] + 1e-6)
                if len(geometric_scores) > self.max_line_candidates:
                    _, top_indices = torch.topk(geometric_scores, self.max_line_candidates)
                    final_lines = junction_filtered_lines[top_indices]
                    final_positions = junction_filtered_positions[top_indices]
                else:
                    final_lines = junction_filtered_lines
                    final_positions = junction_filtered_positions
                
                final_count = len(final_lines)
                debug_stats['final_selected'] += final_count
                print(f"🔄 Geometric fallback: {final_count} lines selected")
            
            # 4. 将筛选结果映射回原始tensor（保持梯度）
            if len(final_lines) > 0:
                pos_h = final_positions // width
                pos_w = final_positions % width
                selected_lines = full_lines[b, :, pos_h, pos_w]
                filtered_lines[b, :, pos_h, pos_w] = selected_lines
        
        # 【新增】打印总体统计信息
        print("\n" + "="*60)
        print("📊 QUALITY FILTERING SUMMARY")
        print("="*60)
        print(f"🔸 Total lines before filtering:     {debug_stats['total_before_filtering']:6d}")
        print(f"🔸 After length filtering:           {debug_stats['after_length_filtering']:6d} ({debug_stats['after_length_filtering']/max(1,debug_stats['total_before_filtering'])*100:.1f}%)")
        print(f"🔸 After junction filtering:         {debug_stats['after_junction_filtering']:6d} ({debug_stats['after_junction_filtering']/max(1,debug_stats['total_before_filtering'])*100:.1f}%)")
        print(f"🔸 After quality assessment:         {debug_stats['after_quality_filtering']:6d} ({debug_stats['after_quality_filtering']/max(1,debug_stats['total_before_filtering'])*100:.1f}%)")
        print(f"🔸 Final selected lines:             {debug_stats['final_selected']:6d} ({debug_stats['final_selected']/max(1,debug_stats['total_before_filtering'])*100:.1f}%)")
        print(f"🔸 Quality filtering ratio:          {debug_stats['final_selected']/max(1,debug_stats['after_junction_filtering'])*100:.1f}%")
        print("="*60)
        
        return filtered_lines

    def forward(self, backbone_feat, stage1_pred, hafm_scale=5.0, return_debug=False, 
                external_modules=None):
        """
        Args:
            backbone_feat: [B, 256, H, W] 骨干特征
            stage1_pred: [B, 9, H, W] Stage1预测结果
            hafm_scale: HAFM解码的scale参数
            external_modules: 外部传入的所有模块和函数
        """
        # 1-2. 解码部分
        md_pred = stage1_pred[:, :3]
        dis_pred = stage1_pred[:, 3:4]
        res_pred = stage1_pred[:, 4:5]
        
        junctions_coords, junctions_confidence = self.simple_junction_decode(stage1_pred)
        full_lines = self.full_hafm_decoding(md_pred, dis_pred, res_pred, scale=hafm_scale)
        
        # 3. 【核心】根据是否传入外部模块选择筛选方式
        if external_modules is not None:
            # 使用外部模块进行完整质量评估
            final_filtered_lines = self.external_quality_guided_line_filtering(
                full_lines, junctions_coords, junctions_confidence, 
                backbone_feat, external_modules
            )
        else:
            # 回退到几何筛选
            final_filtered_lines = self.junction_guided_line_filtering_with_length_and_gradient_detach(
                full_lines, junctions_coords, junctions_confidence
            )
        
        # 4-5. 特征编码和融合
        line_coord_feat = self.line_coord_encoder(final_filtered_lines)
        point_coord_feat = self.point_coord_encoder(junctions_coords)
        
        #-----------------------------------------------------#这上面是prompt模块，下面的是fusion
        junction_input = torch.cat([backbone_feat, line_coord_feat], dim=1)
        junction_enhanced_feat = self.junction_fusion(junction_input) + backbone_feat
        
        line_input = torch.cat([backbone_feat, point_coord_feat], dim=1)
        line_enhanced_feat = self.line_fusion(line_input) + backbone_feat
        
        # 应用轻量级Cross-Attention进行点线特征融合
        final_junction_feat, final_line_feat = self.cross_attention(
            backbone_feat, junction_enhanced_feat, line_enhanced_feat
        )
        
        if return_debug:
            debug_info = self._extract_debug_info(final_filtered_lines, junctions_coords, junctions_confidence, stage1_pred)
            return final_junction_feat, final_line_feat, debug_info
        
        return final_junction_feat, final_line_feat


class PointLineNet(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.point_detector = SuperPoint({})
        for param in self.point_detector.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # [H, W]
        self.conv1a = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # [H/2, W/2]
        self.conv2a = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        self.stack1 = UNet(256, 128, 128, 4)
        self.fc1 = nn.Conv2d(128, 256, kernel_size=1)
        self.score1 = head(256, 9)
        self.stack2 = UNet(128, 128, 128, 4)
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1)
        self.score2 = head(256, 9)
        
        self.prompt_fusion = LightweightCoordinatePromptFusion(feature_channels=256, top_ratio=0.2, max_line_candidates=100)
        self.using_prompt_fusion = True  # 是否使用自适应通道提示融合


    
    def forward(self, image, return_debug=False,external_modules=None):  # 添加 return_debug 参数
        torch.cuda.synchronize()
        t0 = time.time()
# keypoint特征提取
        points = self.point_detector(image[:, :1, ...])
        torch.cuda.synchronize()
        t1 = time.time()

        features = points['features']

# 自定义网络 特征融合：
        x1 = self.relu(self.bn1a(self.conv1a(features[0])))
        x1 = self.relu(self.bn1b(self.conv1b(x1)))          # [B, 64, H, W]

        x2 = self.pool(x1)                                  # [B, 64, H/2, W/2]

        x2 = torch.cat([x2, features[1]], -3)               # [B, 128, H/2, W/2]
        x2 = self.relu(self.bn2a(self.conv2a(x2)))
        x2 = self.relu(self.bn2b(self.conv2b(x2)))          # [B, 128, H/2, W/2]

        x3 = self.pool(x2)                                  # [B, 128, H/4, W/4]
        x3 = torch.cat([x3, features[2]], -3)               # [B, 256, H/4, W/4]


        torch.cuda.synchronize()
        t2 = time.time()

#UNet部分
        x_stack1 = self.stack1(x3)
        x_stack1_ = self.fc1(x_stack1)
        score1 = self.score1(x_stack1_, x_stack1_)
        
        # Stack 2: 第二阶段交叉增强预测
        x_stack2 = self.stack2(x_stack1)
        x_stack2_ = self.fc2(x_stack2)

#fusion模块，输出：junction_feat, line_feat = self.prompt_fusion(x_stack2_, score1, ...)
        # Prompt fusion with external modules
        debug_info = None
        if self.using_prompt_fusion:
            if return_debug:
                junction_feat, line_feat, debug_info = self.prompt_fusion(
                    x_stack2_, score1, return_debug=True, 
                    external_modules=external_modules
                )
            else:
                junction_feat, line_feat = self.prompt_fusion(
                    x_stack2_, score1, 
                    external_modules=external_modules
                )
        else:
            junction_feat = line_feat = x_stack2_

#detection head：输入两个融合对方特征后的feat，输出 9 channels
        # 交叉输入  score2 -> SplitMultitaskHead
        score2 = self.score2(junction_feat, line_feat)
        
        if return_debug:
            return [score2, score1], x_stack2_, debug_info
        
        return [score2, score1], x_stack2_

