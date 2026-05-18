from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)


def compute_rmse(img1, img2):
    mse = F.mse_loss(img1, img2)
    return torch.sqrt(mse)