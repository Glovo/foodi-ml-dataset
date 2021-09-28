
import torch.optim.lr_scheduler


_scheduler = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'step': torch.optim.lr_scheduler.StepLR,
}

def get_scheduler(name, optimizer, **kwargs):
    return _scheduler[name](optimizer=optimizer, **kwargs)
