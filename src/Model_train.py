#!/usr/bin/env python3
import torch
import numpy as np
import time
from tqdm.auto import tqdm
import csv
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GraphDataset import GraphDataset, graph_collate_fn, load_dataset_simple, recursive_to_tensor, recursive_to_tensor_grad
from VEGN_Solver import compute_vem_from_alpha
from VEM_GNN_m2 import VEMGNN, EarlyStopping, GraphDataAugmentation
from check_gradient_flow import check_gradient_flow, ParameterMonitor


def combined_loss(vem_pred, vem_target, device="cpu"):
    mse = nn.MSELoss()
    return mse(vem_pred, vem_target)


def train_epoch(epo, model, dataloader, optimizer, device="cpu", clip_value=1.0):
    """
    训练函数
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    augmenter = GraphDataAugmentation()

    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epo} Training', leave=True, position=0)
    
    for batch_idx, batch in enumerate(pbar):
        # 使用递归函数将整个 batch 移动到设备
        batch = recursive_to_tensor(batch, device)

        X_batch = batch['X']  # 列表
        A_batch = batch['A']  # 列表
        y_batch = batch['y']

        Kc_batch = recursive_to_tensor_grad(batch['Kc'])
        Ks_batch = recursive_to_tensor_grad(batch['Ks'])
        d_batch = recursive_to_tensor_grad(batch['d'])
        nd_batch = batch['nd']
        LM_batch = batch['LM']
        
        batch_start_time = time.time()  # 计时

        # 批量处理：一次性处理整个batch
        optimizer.zero_grad()
        batch_loss = 0.0
        
        vem_pred_list = []
        for i in range(len(X_batch)):
            # 获取当前样本的数据
            X_sample = X_batch[i].requires_grad_(True)
            A_sample = A_batch[i].requires_grad_(True)

            Kc_sample = Kc_batch[i]
            Ks_sample = Ks_batch[i]
            d_sample = d_batch[i]
            nd_sample = nd_batch[i]
            LM_sample = LM_batch[i]

            # 应用数据增强(掩码)
            X_aug, A_aug = augmenter(X_sample, A_sample)

            alpha_list = model(X_aug, A_aug)  # 返回alpha列表
            
            # 定义总刚
            K_global = torch.zeros(len(d_sample), len(d_sample), requires_grad=True).to(device)
            # 求解PDE
            vem_pred = compute_vem_from_alpha(K_global, alpha_list, Kc_sample, Ks_sample, d_sample, nd_sample, LM_sample)
            vem_pred_list.append(vem_pred)

        # 批量计算损失
        vem_preds_tensor = torch.stack([p.squeeze(0) for p in vem_pred_list]).squeeze(1).requires_grad_(True)
        y_tensor = torch.stack([y.unsqueeze(0) for y in y_batch]).squeeze(1).requires_grad_(True)
        loss = combined_loss(vem_preds_tensor, y_tensor, device)
        
        # 一次性反向传播 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value) # 梯度裁剪

        # 每隔n个epoch检查梯度
        if epoch % 5 == 0:
            check_gradient_flow(model, epo, batch_idx, "after BP")
            # 参数监控
            param_monitor.check_parameter_changes(epo, batch_idx)

        optimizer.step()

        batch_loss += loss.item()
        
        average_batch_loss = batch_loss / len(X_batch)
        total_loss += average_batch_loss
        num_batches += 1

        # 更新进度条描述
        batch_time = time.time() - batch_start_time

        pbar.set_postfix({
            'Batch_Loss': f'{average_batch_loss:.4f}',
            'Avg_Loss': f'{total_loss/num_batches:.4f}',
            'Time/Batch': f'{batch_time:.2f}s',
            'Samples': f'{(batch_idx+1)*len(X_batch)}/{len(dataloader.dataset)}'
        })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(epo, model, dataloader, device="cpu"):
    """
    测试函数
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epo} Evaluating', leave=True, position=0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 使用递归函数将整个 batch 移动到设备
            batch = recursive_to_tensor(batch, device)

            X_batch = batch['X']
            A_batch = batch['A']
            y_batch = batch['y']

            Kc_batch = batch['Kc']
            Ks_batch = batch['Ks']
            d_batch = batch['d']
            nd_batch = batch['nd']
            LM_batch = batch['LM']
            
            batch_loss = 0.0
            batch_start_time = time.time()
            
            vem_pred_list = []
            for i in range(len(X_batch)):
                # 获取当前样本的数据
                X_sample = X_batch[i] 
                A_sample = A_batch[i]  
                y = y_batch[i].unsqueeze(0)

                Kc_sample = Kc_batch[i]
                Ks_sample = Ks_batch[i]
                d_sample = d_batch[i]
                nd_sample = nd_batch[i]
                LM_sample = LM_batch[i]
                
                alpha_list = model(X_sample, A_sample)
                
                # 定义总刚
                K_global = torch.zeros(len(d_sample), len(d_sample)).to(device)
                # 求解PDE
                vem_pred = compute_vem_from_alpha(K_global, alpha_list, Kc_sample, Ks_sample, d_sample, nd_sample, LM_sample)
                vem_pred_list.append(vem_pred)
                
            vem_preds_tensor = torch.stack([p.squeeze(0) for p in vem_pred_list]).squeeze(1)
            y_tensor = torch.stack([y.unsqueeze(0) for y in y_batch]).squeeze(1).requires_grad_(True)
            loss = combined_loss(vem_preds_tensor, y_tensor, device)
            
            batch_loss += loss.item()
            
            average_batch_loss = batch_loss / len(X_batch)
            total_loss += average_batch_loss
            num_batches += 1

            # 更新进度条描述
            batch_time = time.time() - batch_start_time

            pbar.set_postfix({
                'Batch_Loss': f'{average_batch_loss:.4f}',
                'Avg_Loss': f'{total_loss/num_batches:.4f}',
                'Time/Batch': f'{batch_time:.2f}s',
                'Samples': f'{(batch_idx+1)*len(X_batch)}/{len(dataloader.dataset)}'
            })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def count_nn_parameter(model):
    p_total = sum(p.numel() for p in model.parameters())
    p_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return p_total, p_trainable


def save_model(model, optimizer, epoch, loss, path):
    """
    保存模型检查点
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_model(model, optimizer, path):
    """
    加载模型检查点
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def write_model(epoches_total, train_loss_list, val_loss_list, lr_list, steptime_list, path):
    """
    将loss写入csv文件
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch','Learn_Late', 'Train_Loss', 'Val_Loss', 'Step_time'])
        
        for epo in range(1, epoches_total):
            train_loss = train_loss_list[epo-1]
            val_loss = val_loss_list[epo-1]
            lr = lr_list[epo-1]
            st = steptime_list[epo-1]
            writer.writerow([epo, lr, train_loss, val_loss, st])


### GNN train function
if __name__ == "__main__":
    """
    GNN神经网络训练主函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 解包数据
    filename = 'D:/bo4_2025/MLcode-test/train4-modify2_NN1_merg/file/dataset_merg_9000_trans_2_rand'  # 后不加 '/' or '.pkl.gz'
    X_list, A_list, y_list, Kc_list, Ks_list, d_list, nd_list, LM_list = load_dataset_simple(filename)
    train_size = int(0.8 * len(X_list))

    # 存储路径
    savepath = './checkpoints/model_9000_trans_2_1'  # 后不加 '/' or '.pkl.gz'

    # 创建数据集
    train_dataset = GraphDataset(
        X_list[:train_size], A_list[:train_size], y_list[:train_size], 
        Kc_list[:train_size], Ks_list[:train_size], d_list[:train_size],
        nd_list[:train_size], LM_list[:train_size]
    )
    val_dataset = GraphDataset(
        X_list[train_size:], A_list[train_size:], y_list[train_size:],
        Kc_list[train_size:], Ks_list[train_size:], d_list[train_size:],
        nd_list[train_size:], LM_list[train_size:]
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )

    # 模型
    model = VEMGNN().to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # 初始化自适应学习率
    scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',           # 监控验证损失最小化
    factor=0.5,           # 学习率减半
    patience=3,           # n个epoch没有改善就降低学习率
    min_lr=1e-6           # 最小学习率
    )
    # 初始化早停器
    early_stopping = EarlyStopping(
        patience=10,              # 连续n个epoch验证损失没有改善则停止
        min_delta=1e-6,           # 最小改善量
        restore_best_weights=True # 恢复最佳权重
    )

    # 超参数统计
    p_total, p_trainable = count_nn_parameter(model)
    print(f'Model total parameters = {p_total}, trainable parameters = {p_trainable}')

    # 训练
    epoches_total = 101
    train_loss_list = [None]*(epoches_total-1)  # 记录训练损失
    val_loss_list = [None]*(epoches_total-1)    # 记录验证损失
    lr_list = [None]*(epoches_total-1)  # 记录学习率
    steptime_list = [None]*(epoches_total-1)  # 记录单步用时

    # 初始化监控器
    param_monitor = ParameterMonitor(model)

    start_time = time.time() # 计时开始
    print(f'Start train: time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}, total epoch = {epoches_total-1}\n')
    
    for epoch in range(1, epoches_total):
        start_step_time = time.time() # 该步计时开始
        # training
        train_loss = train_epoch(epoch, model, train_loader, optimizer, device)
        train_loss_list[epoch-1] = train_loss
        # validating
        val_loss = evaluate(epoch, model, val_loader, device)
        val_loss_list[epoch-1] = val_loss

        # 根据验证损失调整学习率
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        lr_list[epoch-1] = current_lr
        steptime_list[epoch-1] = time.time() - start_step_time
        print(f"\nEpoch {epoch}: LR = {current_lr:.2e}, train_loss={train_loss:.8f}, val_loss={val_loss:.8f}\n")

        # 检查早停条件
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # 隔固定epoch存储一遍model
        if epoch % 2 == 0:
            save_model(model, optimizer, epoches_total, val_loss, f'{savepath}.pth')  # 保存模型
            write_model(epoches_total, train_loss_list, val_loss_list, lr_list, steptime_list, f'{savepath}.csv')  # 写入监控数据
    
    end_time = time.time()
    print(f"\nTraning Time={(end_time - start_time):.6f} s")

    # 后处理
    save_model(model, optimizer, epoches_total, val_loss, f'{savepath}.pth')  # 保存模型
    write_model(epoches_total, train_loss_list, val_loss_list, lr_list, steptime_list, f'{savepath}.csv')  # 写入监控数据
