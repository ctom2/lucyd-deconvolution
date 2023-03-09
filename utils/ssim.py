import torch

def get_SSIM(gt, pred):
    mean_pred = torch.mean(pred)
    mean_gt = torch.mean(gt)
    sigma_pred = torch.mean(torch.square(pred - mean_pred))
    sigma_gt = torch.mean(torch.square(gt - mean_gt))
    sigma_cross = torch.mean((pred - mean_pred) * (gt - mean_gt))

    SSIM_1 = 2 * mean_pred * mean_gt + 1e-4
    SSIM_2 = 2 * sigma_cross + 9e-4
    SSIM_3 = torch.square(mean_pred) + torch.square(mean_gt) + 1e-4
    SSIM_4 = sigma_pred + sigma_gt + 9e-4
    SSIM = (SSIM_1*SSIM_2)/(SSIM_3*SSIM_4)

    return SSIM