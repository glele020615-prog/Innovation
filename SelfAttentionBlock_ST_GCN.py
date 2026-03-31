import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# Transformer Layer for Temporal and Spatial Attention
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.7):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Temporal Attention Block
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, num_time_points, num_nodes, d_model=32, nhead=8, dropout=0.7):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.proj = nn.Linear(in_channels, d_model)
        self.transformer = TransformerLayer(d_model, nhead, dropout=dropout)
        self.conv = nn.Conv2d(d_model, d_model // 2, kernel_size=(3, 1), padding=(1, 0), stride=1)

    def forward(self, x):
        N, C, T, V = x.size()  # [N, C, T, V], e.g., [16, 1, 130, 116]
        x = x.permute(0, 2, 3, 1)  # [N, T, V, C]
        x = self.proj(x)  # [N, T, V, d_model]
        x = x.permute(2, 0, 1, 3)  # [V, N, T, d_model]
        x = x.reshape(V * N, T, self.d_model)  # [V*N, T, d_model]
        x = self.transformer(x)  # [V*N, T, d_model]
        x = x.view(V, N, T, self.d_model)  # [V, N, T, d_model]
        x = x.permute(1, 2, 0, 3)  # [N, T, V, d_model]
        x = x.permute(0, 3, 1, 2)  # [N, d_model, T, V]
        x = self.conv(x)  # [N, d_model//2, T, V], e.g., [16, 16, 130, 116]
        return x

# Spatial Attention Block
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, num_time_points, num_nodes, d_model=32, nhead=8, dropout=0.7):
        super().__init__()
        self.num_time_points = num_time_points
        self.d_model = d_model
        self.proj = nn.Linear(in_channels, d_model)
        self.transformer = TransformerLayer(d_model, nhead, dropout=dropout)
        self.conv = nn.Conv2d(d_model, d_model // 2, kernel_size=(1, 3), padding=(0, 1), stride=1)

    def forward(self, x):
        N, C, T, V = x.size()  # [N, C, T, V], e.g., [16, 1, 130, 116]
        x = x.permute(0, 3, 2, 1)  # [N, V, T, C]
        x = self.proj(x)  # [N, V, T, d_model]
        x = x.permute(2, 0, 1, 3)  # [T, N, V, d_model]
        x = x.reshape(T * N, V, self.d_model)  # [T*N, V, d_model]
        x = self.transformer(x)  # [T*N, V, d_model]
        x = x.view(T, N, V, self.d_model)  # [T, N, V, d_model]
        x = x.permute(1, 2, 0, 3)  # [N, V, T, d_model]
        x = x.permute(0, 3, 2, 1)  # [N, d_model, T, V]
        x = self.conv(x)  # [N, d_model//2, T, V], e.g., [16, 16, 130, 116]
        return x

# Global Attention
class GlobalAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, reduction_factor=4, num_heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.reduction_factor = reduction_factor
        self.pool = nn.AdaptiveAvgPool2d((None, num_nodes // reduction_factor))
        self.global_attn = nn.MultiheadAttention(in_channels, num_heads=num_heads)
        self.global_norm = nn.LayerNorm(in_channels)
        self.out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        N, C, T, V = x.size()  # [N, C, T, V], e.g., [16, 32, 130, 116]
        x_global = self.pool(x)  # [N, C, T, V//reduction_factor], e.g., [16, 32, 130, 29]
        V_red = x_global.size(-1)
        x_global = x_global.view(N, C, T * V_red).permute(2, 0, 1)  # [T*V_red, N, C]
        x_global, _ = self.global_attn(x_global, x_global, x_global)  # [T*V_red, N, C]
        x_global = self.global_norm(x_global)
        x_global = x_global.permute(1, 2, 0).view(N, C, T, V_red)  # [N, C, T, V_red]
        # Interpolate to restore original V
        x_global = F.interpolate(x_global, size=(T, V), mode='bilinear', align_corners=False)  # [N, C, T, V]
        x_out = self.out(x_global.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [N, C, T, V]
        x = x + x_out  # 残差连接
        return x


# ST-GCN Components
class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_stride=1, dropout=0.5, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GCN(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (t_stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (t_stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(t_stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, _ = self.gcn(x, A)
        x = self.tcn(x) + res
        x = self.bn(x)
        x = self.relu(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=3, t_stride=1, t_padding=1, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        N, C, T, V = x.size()
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        A = A.squeeze(1)
        x = torch.einsum('nkctv,nvw->nctw', (x, A))
        return x.contiguous(), A

# Updated Model with Spatio-Temporal Attention
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, edge_importance_weighting, temporal_kernel_size, kernel_gcn, num_nodes=116, num_time_points=130, d_model=32, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        # Spatio-Temporal Attention Blocks
        self.temporal_block = TemporalBlock(in_channels, num_time_points, num_nodes, d_model=d_model)
        self.spatial_block = SpatialBlock(in_channels, num_time_points, num_nodes, d_model=d_model)
        # Global Attention for Feature Fusion
        self.fusion = GlobalAttention(d_model, num_nodes, reduction_factor=2, num_heads=4)
        # ST-GCN Layers
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(d_model, 128, (temporal_kernel_size, kernel_gcn), residual=True),
            st_gcn(128, 64, (temporal_kernel_size, kernel_gcn), residual=True),
            st_gcn(64, out_channels, (temporal_kernel_size, kernel_gcn), residual=True),
        ))
        self.pool = nn.AdaptiveAvgPool2d((1, num_nodes))
        self.linear = nn.Linear(out_channels * num_nodes, num_class)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones((1, num_nodes, num_nodes))) for _ in range(len(self.st_gcn_networks))
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x, A):
        N, C, T, V = x.size()  # [N, C, T, V], e.g., [16, 1, 130, 116]
        #print(f"Input x shape: {x.shape}, A shape: {A.shape}")
        A = A.squeeze(1)  # [N, V, V], e.g., [16, 116, 116]
        Dn = torch.zeros((N, V, V), dtype=torch.float32, device=A.device)
        Dl = torch.sum(A, dim=1)  # [N, V]
        for n in range(N):
            for i in range(V):
                if Dl[n, i] > 0:
                    Dn[n, i, i] = (Dl[n, i] + 1) ** (-0.5)
        A_static = torch.bmm(torch.bmm(Dn, A + torch.eye(V, device=A.device)), Dn).unsqueeze(1)  # [N, 1, V, V]

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)  # [N, V*C, T]
        x = self.data_bn(x).view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # [N, C, T, V]
        #print(f"After data_bn shape: {x.shape}")

        # Spatio-Temporal Attention
        temp_features = self.temporal_block(x)  # [N, d_model//2, T, V]
        #print(f"Temporal features shape: {temp_features.shape}")
        spatial_features = self.spatial_block(x)  # [N, d_model//2, T, V]
        #print(f"Spatial features shape: {spatial_features.shape}")
        fused_features = torch.cat([temp_features, spatial_features], dim=1)  # [N, d_model, T, V]
        #print(f"Fused features shape: {fused_features.shape}")
        fused_features = self.fusion(fused_features)  # [N, d_model, T, V]
        #print(f"After fusion shape: {fused_features.shape}")

        # ST-GCN Layers
        for i, gcn in enumerate(self.st_gcn_networks):
            tmp_abs_edge = torch.abs(self.edge_importance[i]) if isinstance(self.edge_importance, nn.ParameterList) else 1
            adj = A_static * ((tmp_abs_edge / 2 + tmp_abs_edge.transpose(-1, -2) / 2))
            fused_features = gcn(fused_features, adj)
            #print(f"After st_gcn layer {i+1} shape: {fused_features.shape}")

        x = self.pool(fused_features)  # [N, out_channels, 1, V]
        x = x.view(x.size(0), -1)  # [N, out_channels*V]
        x = self.linear(x)  # [N, num_class]
        x = F.log_softmax(x, dim=1)
        #print(f"Output shape: {x.shape}")
        return x

# Model Instantiation and Summary
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(
        in_channels=1,
        out_channels=32,
        num_class=2,
        edge_importance_weighting=True,
        temporal_kernel_size=7,
        kernel_gcn=3,
        num_nodes=116,
        num_time_points=130,
        d_model=32
    ).to(device)
    summary(model, input_size=[(8, 1, 130, 116), (8, 1, 116, 116)])