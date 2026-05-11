import torch
from tqdm import tqdm
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import wandb
import numpy as np
import os
import sys
import torch.nn.functional as F

from util.util import get_logger, mkdirs, save_images, compute_ssim, compute_rmse, compute_psnr2D

# ===== SSIM 相关 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
ssim_path = os.path.join(current_dir, 'pytorch_ssim_package')
if ssim_path not in sys.path:
    sys.path.insert(0, ssim_path)

from pytorch_ssim import ssim3D

# ===== 反归一化（你已写好，保留）=====
def de_normalize(x, min_value=-1000, max_value=2000):
    return ((x + 1) / 2) * (max_value - min_value) + min_value


# ============================================================
#  新增：Gradient / Structure Loss
# ============================================================
def gradient_loss(pred, gt):
    """
    pred, gt: (B, 1, 1, H, W) or (B, 1, H, W)
    return: scalar
    """
    # squeeze depth dimension if exists
    if pred.dim() == 5:
        pred = pred.squeeze(2)  # (B,1,H,W)
        gt = gt.squeeze(2)

    # x-direction gradient
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_g = gt[:, :, :, 1:] - gt[:, :, :, :-1]

    # y-direction gradient
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_g = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    loss_dx = F.l1_loss(dx_p, dx_g)
    loss_dy = F.l1_loss(dy_p, dy_g)

    return loss_dx + loss_dy


# ============================================================
# 主训练函数
# ============================================================
def train(opt, model, optimizer, lr_scheduler, loss_fn, trainloader, testloader, device):

    if opt.use_wandb:
        wandb.init(project='litformer_review', name=opt.name)
        wandb.watch(model)

    train_logger = get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'train.log'))
    save_images_root = os.path.join('./results', opt.name)
    mkdirs(save_images_root)
    train_logger.info('start training!')

    best_val_psnr2d = float('-inf')
    train_total_iters = 0
    val_total_iters = 0

    Lambda = 2.0          # SSIM 权重（你原来就有）
    Gamma = 0.1          # Gradient loss 权重（新加，论文级安全值）

    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch', leave=True):

        # ============================
        # Training Phase
        # ============================
        model.train()
        running_metrics = {
            'loss': 0,
            'psnr3d': 0,
            'psnr2d': 0,
            'ssim3d': 0,
            'ssim2d': 0,
            'rmse': 0
        }

        train_progress = tqdm(trainloader,
                              desc=f'Train Epoch {epoch + 1}/{opt.epochs}',
                              leave=False,
                              unit='batch',
                              ncols=100)

        for x, y in train_progress:
            x, y = x.to(device), y.to(device)

            # Forward
            y_pred = model(x)

            # ===== Loss 计算（核心修改点）=====
            loss_rec = loss_fn(y_pred, y)
            loss_ssim = Lambda * (1 - compute_ssim(y_pred, y))
            loss_grad = Gamma * gradient_loss(y_pred, y)

            train_loss = loss_rec + loss_ssim + loss_grad

            # Metrics
            train_psnr3d = peak_signal_noise_ratio(y_pred, y)
            train_psnr2d = compute_psnr2D(y_pred, y)
            train_ssim3d = ssim3D(y_pred, y)
            train_ssim2d = compute_ssim(y_pred, y)
            train_rmse = compute_rmse(y_pred, y)

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update running metrics
            running_metrics['loss'] += train_loss.item()
            running_metrics['psnr3d'] += train_psnr3d.item()
            running_metrics['psnr2d'] += train_psnr2d.item()
            running_metrics['ssim3d'] += train_ssim3d.item()
            running_metrics['ssim2d'] += train_ssim2d.item()
            running_metrics['rmse'] += train_rmse.item()

            train_total_iters += 1

            # Print & WandB
            if train_total_iters % 100 == 0:
                message = (
                    f'(epoch: {epoch}, iters: {train_total_iters}, '
                    f'loss: {train_loss.item():.4f}, '
                    f'psnr3d: {train_psnr3d.item():.4f}, '
                    f'psnr2d: {train_psnr2d.item():.4f}, '
                    f'ssim3d: {train_ssim3d.item():.4f}, '
                    f'ssim2d: {train_ssim2d.item():.4f}, '
                    f'rmse: {train_rmse.item():.6f})'
                )
                print(message)

                if opt.use_wandb:
                    wandb.log({
                        "train/loss": train_loss.item(),
                        "train/psnr3d": train_psnr3d.item(),
                        "train/psnr2d": train_psnr2d.item(),
                        "train/ssim3d": train_ssim3d.item(),
                        "train/ssim2d": train_ssim2d.item(),
                        "train/rmse": train_rmse.item(),
                        "train/loss_rec": loss_rec.item(),
                        "train/loss_grad": loss_grad.item()
                    })

        # Epoch 平均
        train_length = len(trainloader.dataset)
        epoch_metrics = {k: v / train_length * opt.train_batch_size
                         for k, v in running_metrics.items()}

        train_logger.info(
            f'Epoch: [{epoch}/{opt.epochs}], '
            f'loss: {epoch_metrics["loss"]:.6f}, '
            f'psnr3d: {epoch_metrics["psnr3d"]:.4f}, '
            f'psnr2d: {epoch_metrics["psnr2d"]:.4f}, '
            f'ssim3d: {epoch_metrics["ssim3d"]:.4f}, '
            f'ssim2d: {epoch_metrics["ssim2d"]:.4f}, '
            f'rmse: {epoch_metrics["rmse"]:.6f}'
        )

        # ============================
        # Validation Phase
        # ============================
        model.eval()
        val_metrics = {
            'loss': 0,
            'psnr3d': 0,
            'psnr2d': 0,
            'ssim3d': 0,
            'ssim2d': 0,
            'rmse': 0,
            'all_psnr3d': [],
            'all_psnr2d': [],
            'all_ssim3d': [],
            'all_ssim2d': [],
            'all_rmse': []
        }

        with torch.no_grad():
            for x, y in tqdm(testloader, desc='Validation'):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)

                # ===== Validation Loss（同样加入 gradient）=====
                loss_rec = loss_fn(y_pred, y)
                loss_ssim = Lambda * (1 - compute_ssim(y_pred, y))
                loss_grad = Gamma * gradient_loss(y_pred, y)

                test_loss = loss_rec + loss_ssim + loss_grad

                # Metrics
                test_psnr3d = peak_signal_noise_ratio(y_pred, y)
                test_psnr2d = compute_psnr2D(y_pred, y)
                test_ssim3d = ssim3D(y_pred, y)
                test_ssim2d = compute_ssim(y_pred, y)
                test_rmse = compute_rmse(y_pred, y)

                val_metrics['loss'] += test_loss.item()
                val_metrics['psnr3d'] += test_psnr3d.item()
                val_metrics['psnr2d'] += test_psnr2d.item()
                val_metrics['ssim3d'] += test_ssim3d.item()
                val_metrics['ssim2d'] += test_ssim2d.item()
                val_metrics['rmse'] += test_rmse.item()

                val_metrics['all_psnr3d'].append(test_psnr3d.item())
                val_metrics['all_psnr2d'].append(test_psnr2d.item())
                val_metrics['all_ssim3d'].append(test_ssim3d.item())
                val_metrics['all_ssim2d'].append(test_ssim2d.item())
                val_metrics['all_rmse'].append(test_rmse.item())

        # 平均 + 方差
        val_length = len(testloader.dataset)
        avg_val_metrics = {
            'loss': val_metrics['loss'] / val_length,
            'psnr3d': val_metrics['psnr3d'] / val_length,
            'psnr2d': val_metrics['psnr2d'] / val_length,
            'ssim3d': val_metrics['ssim3d'] / val_length,
            'ssim2d': val_metrics['ssim2d'] / val_length,
            'rmse': val_metrics['rmse'] / val_length
        }

        std_metrics = {
            'psnr3d': np.std(val_metrics['all_psnr3d']),
            'psnr2d': np.std(val_metrics['all_psnr2d']),
            'ssim3d': np.std(val_metrics['all_ssim3d']),
            'ssim2d': np.std(val_metrics['all_ssim2d']),
            'rmse': np.std(val_metrics['all_rmse'])
        }

        train_logger.info(
            f'Val Epoch: [{epoch}/{opt.epochs}], '
            f'loss: {avg_val_metrics["loss"]:.6f}, '
            f'psnr3d: {avg_val_metrics["psnr3d"]:.4f}±{std_metrics["psnr3d"]:.4f}, '
            f'psnr2d: {avg_val_metrics["psnr2d"]:.4f}±{std_metrics["psnr2d"]:.4f}, '
            f'ssim3d: {avg_val_metrics["ssim3d"]:.4f}±{std_metrics["ssim3d"]:.4f}, '
            f'ssim2d: {avg_val_metrics["ssim2d"]:.4f}±{std_metrics["ssim2d"]:.4f}, '
            f'rmse: {avg_val_metrics["rmse"]:.6f}±{std_metrics["rmse"]:.6f}'
        )

        # LR step
        lr_scheduler.step()

        # WandB epoch log
        if opt.use_wandb:
            wandb.log({
                "epoch/train_loss": epoch_metrics["loss"],
                "epoch/train_psnr2d": epoch_metrics["psnr2d"],
                "epoch/val_loss": avg_val_metrics["loss"],
                "epoch/val_psnr2d": avg_val_metrics["psnr2d"],
                "epoch": epoch
            })

        # Save best model by PSNR2D
        if avg_val_metrics['psnr2d'] > best_val_psnr2d:
            best_val_psnr2d = avg_val_metrics['psnr2d']
            state_dict = model.module.state_dict() if len(opt.gpu_ids) > 1 else model.state_dict()
            torch.save(
                state_dict,
                os.path.join(opt.checkpoints_dir, opt.name, 'best_val_psnr2d_model.pth')
            )
            train_logger.info(f' Best Val PSNR2D model saved at epoch {epoch}: {best_val_psnr2d:.4f}')

    train_logger.info('finish training!')
