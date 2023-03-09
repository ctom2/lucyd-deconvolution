import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from utils.ssim import get_SSIM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_dataloader, val_dataloader):
    epochs = 200

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    psnr = PeakSignalNoiseRatio().to(device)


    for epoch in range(epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        model.train()
        train_loss = []
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            
            y_hat, y_k, update = model(x.float())

            loss = mse_loss(y_hat.float(), y.float()) - torch.log((1+get_SSIM(y, y_hat))/2)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

        print('train loss: {}'.format(np.mean(np.array(train_loss))))

        if (epoch % 5 == 0) or (epoch + 1 == epochs):
            model.eval()
            val_loss = []
            val_ssim = []
            val_psnr = []
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)

                y_hat, y_k, update = model(x.float())

                loss = mse_loss(y_hat.float(), y.float()) - torch.log((1+get_SSIM(y, y_hat))/2)

                val_loss.append(loss.item())
                val_ssim.append(get_SSIM(y, y_hat).item())
                val_psnr.append(psnr(y, y_hat).item())

            print('testing loss: {}'.format(np.mean(np.array(val_loss))))
            print('testing ssim: {} +- {}'.format(np.round(np.mean(np.array(val_ssim)), 5), np.round(np.std(np.array(val_ssim)), 5)))
            print('testing psnr: {} +- {}'.format(np.round(np.mean(np.array(val_psnr)), 5), np.round(np.std(np.array(val_psnr)), 5)))

    return model