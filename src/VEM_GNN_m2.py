#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, add_self_loops
import numpy as np

class VEMGNN(nn.Module):
    """
    图神经网络
    """
    def __init__(self, hidden_dim=64, feat_dim=4, aggre_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.aggre_layers = aggre_layers
        self.dropout = dropout

        self.input_projectors = nn.ModuleDict()

        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 邻接节点聚合处理器
        self.processor = PyGAggregationProcessor(
            hidden_dim=hidden_dim,
            num_layers=aggre_layers,
            dropout=dropout
        )

        # 节点解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()
        )


    def forward(self, X, A):
        """
        前向传播
        
        参数:
            X: 节点特征列表, (nel, feat_dim)
            A: 邻接矩阵, (nel, nel)
        """
        # 强制类型转换
        X = X.float()
        A = (A > 0).long()

        # 计算特征维度
        total_nodes, feat_dim = X.shape
        
        h = self.node_encoder(X) # 编码

        h = self.processor(h, A) # 聚合

        alpha = self.node_decoder(h)  # 解码 (total_nodes, 1)
        # alpha = 0.01 + (2.0 - 0.01) * alpha
        
        return alpha  # 返回alpha


class PyGAggregationProcessor(nn.Module):
    """
    使用PyG MessagePassing的邻接节点聚合处理器
    """
    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多层PyG聚合层
        self.aggregation_layers = nn.ModuleList([
            GraphConv(hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # 残差连接和归一化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, node_features, adjacency_matrix):
        """
        使用PyG进行邻接节点加法聚合
        """
        # 将稠密邻接矩阵转换为PyG需要的edge_index格式
        edge_index, edge_weight = dense_to_sparse(adjacency_matrix)
        
        h = node_features
        
        for i in range(self.num_layers):
            # 保存残差连接
            residual = h
            
            # 使用PyG层进行邻接节点聚合
            h = self.aggregation_layers[i](h, edge_index)
            
            # 残差连接和归一化
            h = h + residual
            h = self.norm_layers[i](h)
            
            # 最后一层不使用ReLU
            if i < self.num_layers - 1:
                h = F.relu(h)
        
        return h


class GraphConv(MessagePassing):
    """使用PyG MessagePassing实现的简单图卷积层[citation:6][citation:9]"""
    
    def __init__(self, hidden_dim, dropout=0.1, aggr='max'):
        super().__init__(aggr=aggr)  # 聚合方式：add, mean, max
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # 线性变换层
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        """
        PyG MessagePassing前向传播
        """
        # 添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换
        x = self.lin(x)
        
        # 开始消息传播
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """
        定义消息函数：从源节点j到目标节点i的消息
        """
        return x_j
    
    def update(self, aggr_out):
        """
        更新节点特征：应用dropout
        """
        return self.dropout(aggr_out)


class EarlyStopping:
    """
    早停类，用于在验证损失不再改善时停止训练
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        参数:
        patience: 在验证损失不再改善后等待的epoch数
        min_delta: 被视为改善的最小变化量
        restore_best_weights: 是否在早停时恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            
    def restore_best_model(self, model):
        """恢复最佳模型权重"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Restored model weights from the best checkpoint.")


class GraphDataAugmentation:
    def __init__(self, drop_edge_rate=0.1, mask_node_rate=0.05):
        self.drop_edge_rate = drop_edge_rate
        self.mask_node_rate = mask_node_rate
    
    def drop_edges(self, edge_index):
        """随机删除边"""
        num_edges = edge_index.size(1)
        num_remove = int(num_edges * self.drop_edge_rate)
        
        remove_indices = torch.randperm(num_edges)[:num_remove]
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[remove_indices] = False
        
        return edge_index[:, keep_mask]
    
    def mask_nodes(self, x):
        """随机掩码节点特征"""
        mask = torch.rand(x.size(0)) > self.mask_node_rate
        x_masked = x.clone()
        x_masked[~mask] = 0
        return x_masked
    
    def __call__(self, x, edge_index):
        x_aug = self.mask_nodes(x)
        edge_index_aug = self.drop_edges(edge_index)
        return x_aug, edge_index_aug
