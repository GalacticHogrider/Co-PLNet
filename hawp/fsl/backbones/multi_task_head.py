import torch
import torch.nn as nn
class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))
        print(f"input_channels: {input_channels}, num_class: {num_class}, head_size: {head_size}")
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)
    

class SplitMultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(SplitMultitaskHead, self).__init__()
        
        m = int(input_channels / 4)
        
        # 根据head_size分组：[[3], [1], [1], [2], [2]]
        # 线段组：前3个head (3+1+1=5通道)  
        # 交点组：后2个head (2+2=4通道)
        
        flat_head_size = sum(head_size, [])  # [3, 1, 1, 2, 2]
        
        # 线段相关的heads (前3个: md_pred, dis_pred, res_pred)
        self.line_heads = nn.ModuleList()
        for i in range(3):  # 前3个head
            output_channels = flat_head_size[i]
            self.line_heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        
        # 交点相关的heads (后2个: jloc_pred, joff_pred)
        self.junction_heads = nn.ModuleList()
        for i in range(3, 5):  # 后2个head
            output_channels = flat_head_size[i]
            self.junction_heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        
        assert num_class == sum(flat_head_size)
        print(f"input_channels: {input_channels}, num_class: {num_class}, head_size: {head_size}")
        print(f"Line heads: {len(self.line_heads)}, Junction heads: {len(self.junction_heads)}")
        
    def forward(self, PointFusedFeature, LineFusedFeature):
        """
        PointFusedFeature: [B, C, H, W] - 用交点prompt增强的特征 → 用于线段预测
        LineFusedFeature: [B, C, H, W] - 用线段prompt增强的特征 → 用于交点预测
        """
        
        # 线段预测：使用交点增强的特征
        line_outputs = []
        for head in self.line_heads:
            line_outputs.append(head(PointFusedFeature))
        
        # 交点预测：使用线段增强的特征  
        junction_outputs = []
        for head in self.junction_heads:
            junction_outputs.append(head(LineFusedFeature))
        
        # 按原始顺序拼接：[md, dis, res, jloc, joff]
        all_outputs = line_outputs + junction_outputs
        
        return torch.cat(all_outputs, dim=1)


class AngleDistanceHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(AngleDistanceHead, self).__init__()

        m = int(input_channels/4)

        heads = []
        for output_channels in sum(head_size, []):
            if output_channels != 2:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
            else:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        CosineSineLayer(m)
                    )
                )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)