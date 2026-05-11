import os
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio

from util.util import get_logger, make_dir, compute_ssim, compute_rmse


# ===============================
# Gradient Loss
# ===============================
def gradient_loss(pred, target):
    """
    pred, target: [B, C, H, W] or [B, C, D, H, W]
    """
    def gradient(x):
        dy = x[..., 1:, :] - x[..., :-1, :]
        dx = x[..., :, 1:] - x[..., :, :-1]
        return dx, dy

    dx_p, dy_p = gradient(pred)
    dx_t, dy_t = gradient(target)

    return torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))


# ===============================
# Mean HU Error (windowed)
# ===============================
def mean_value(y_pred, y):
    """
    Compute mean absolute error in HU window
    Assumes data normalized to [0,1]
    """
    y_pred_hu = y_pred * 3000 - 1000
    y_hu = y * 3000 - 1000

    y_pred_hu = torch.clamp(y_pred_hu, -160, 240)
    y_hu = torch.clamp(y_hu, -160, 240)

    return torch.mean(torch.abs(y_pred_hu - y_hu)).item()


# ===============================
# Save helper
# ===============================
def save_npy(save_dir, name, tensor):
    """
    tensor: torch.Tensor
    saved as numpy, squeezed to [H,W] or [D,H,W]
    """
    arr = tensor.detach().cpu().numpy()
    arr = np.squeeze(arr)
    np.save(os.path.join(save_dir, name), arr)


# ===============================
# Test Function
# ===============================
def test(opt, model, loss_fn, testloader, device):
    """
    Standard reproducible testing pipeline
    Save: input / pred / gt as .npy
    """

    # ---------- directories ----------
    result_root = f'./test_results/{opt.name}'
    save_npy_dir = os.path.join(result_root, f'{opt.phase}-npy')

    make_dir(result_root)
    make_dir(save_npy_dir)

    # ---------- logger ----------
    logger = get_logger(os.path.join(result_root, 'test.log'))
    logger.info('========== Start Testing ==========')

    # ---------- statistics ----------
    psnr_list, ssim_list, rmse_list, mean_list, grad_list = [], [], [], [], []
    total_loss = 0.0

    Lambda = 2.0
    model.eval()

    iters = 0

    with torch.no_grad():
        for batch in tqdm(testloader, desc='Testing'):

            # -----------------------------
            # dataset output
            # -----------------------------
            if len(batch) == 3:
                x, y, name = batch
                name = name[0]
            else:
                x, y = batch
                name = f'{iters:04d}'

            x, y = x.to(device), y.to(device)

            # -----------------------------
            # forward
            # -----------------------------
            y_pred = model(x)

            # -----------------------------
            # loss
            # -----------------------------
            base_loss = loss_fn(y_pred, y)
            grad_loss = gradient_loss(y_pred, y)
            loss = base_loss + Lambda * grad_loss

            # -----------------------------
            # metrics
            # -----------------------------
            psnr = peak_signal_noise_ratio(y_pred, y).item()
            ssim = compute_ssim(y_pred, y).item()
            rmse = compute_rmse(y_pred, y).item()
            mean_err = mean_value(y_pred, y)

            # -----------------------------
            # record
            # -----------------------------
            total_loss += loss.item()
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            rmse_list.append(rmse)
            mean_list.append(mean_err)
            grad_list.append(grad_loss.item())

            # -----------------------------
            # save npy (核心)
            # -----------------------------
            save_npy(save_npy_dir, f'{name}_input.npy', x)
            save_npy(save_npy_dir, f'{name}_pred.npy', y_pred)
            save_npy(save_npy_dir, f'{name}_gt.npy', y)

            # -----------------------------
            # log
            # -----------------------------
            logger.info(
                f'[{name}] '
                f'Loss: {loss.item():.6f} | '
                f'PSNR: {psnr:.4f} | '
                f'SSIM: {ssim:.4f} | '
                f'RMSE: {rmse:.6f} | '
                f'MEAN: {mean_err:.4f}'
            )

            iters += 1

    # ---------- summary ----------
    logger.info('========== Test Summary ==========')
    logger.info(f'Avg Loss: {total_loss / iters:.6f}')
    logger.info(f'Avg PSNR: {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}')
    logger.info(f'Avg SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}')
    logger.info(f'Avg RMSE: {np.mean(rmse_list):.6f}')
    logger.info(f'Avg Mean HU Error: {np.mean(mean_list):.4f}')
    logger.info(f'Avg Grad Loss: {np.mean(grad_list):.6f}')
    logger.info('========== Finish ==========')


