import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from utils.ssim import get_SSIM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader):
    model.eval()

    psnr = PeakSignalNoiseRatio().to(device)

    val_ssim = []
    val_psnr = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_hat, y_k, update = model(x.float())

        val_ssim.append(get_SSIM(y, y_hat).item())
        val_psnr.append(psnr(y, y_hat).item())

    print('testing ssim: {} +- {}'.format(np.round(np.mean(np.array(val_ssim)), 5), np.round(np.std(np.array(val_ssim)), 5)))
    print('testing psnr: {} +- {}'.format(np.round(np.mean(np.array(val_psnr)), 5), np.round(np.std(np.array(val_psnr)), 5)))