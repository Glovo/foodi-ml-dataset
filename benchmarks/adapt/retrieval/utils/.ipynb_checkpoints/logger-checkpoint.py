from collections import OrderedDict
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return f'{self.val:6g}'
        # for stats
        return f'{self.val:.3f} ({self.avg:.3f})'


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """
            Concatenate the meters in one log line
        """
        s = ''
        for k, v in self.meters.items():
            s += f'{k.title()} {v}\t'
        return s.rstrip()

    def tb_log(self, tb_logger, prefix='data/', step=None):
        """
            Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, step)

    def update_dict(
        self, val_metrics,
    ):

        for metric_name, metric_val in val_metrics.items():
            try:
                v = metric_val.item()
            except AttributeError:
                v = metric_val

            self.update(
                k=f'{metric_name}', v=v, n=0
            )


def create_logger(level='info'):

    level = eval(f'logging.{level.upper()}')

    logging.basicConfig(
        format='%(asctime)s - [%(levelname)-8s] - %(message)s',
        level=level
    )

    logger = logging.getLogger(__name__)
    return logger


def get_logger():
    logger = logging.getLogger(__name__)
    return logger


def tb_log_dict(tb_writer, data_dict, iteration, prefix=''):
    for k, v in data_dict.items():
        tb_writer.add_scalar(f'{prefix}/{k}', v, iteration)


def log_param_histograms(model, tb_writer, iteration):
    for k, p in model.named_parameters():
        tb_writer.add_histogram(
            f'params/{k}',
            p.data,
            iteration,
        )

def log_grad_norm(model, tb_writer, iteration, reduce=sum):

    grads = []
    for k, p in model.named_parameters():
        if p.grad is None:
            continue
        tb_writer.add_scalar(
            f'grads/{k}',
            p.grad.data.norm(2).item(),
            iteration
        )
        grads.append(p.grad.data.norm(2).item())
    return reduce(grads)


def print_log_param_stats(model, iteration):

    print('Iter s{}'.format(iteration))
    for k, v in model.txt_enc.named_parameters():
        print('{:35s}: {:8.5f}, {:8.5f}, {:8.5f}, {:8.5f}'.format(
            k, v.data.cpu().min().numpy(),
            v.data.cpu().mean().numpy(),
            v.data.cpu().max().numpy(),
            v.data.cpu().std().numpy(),
        ))
    for k, v in model.img_enc.named_parameters():
        print('{:35s}: {:8.5f}, {:8.5f}, {:8.5f}, {:8.5f}'.format(                k, v.data.cpu().min().numpy(),
            v.data.cpu().mean().numpy(),
            v.data.cpu().max().numpy(),
            v.data.cpu().std().numpy(),
        ))
    for k, p in model.txt_enc.named_parameters():
        if p.grad is None:
            continue
        print('{:35s}: {:8.5f}'.format(k, p.grad.data.norm(2).item(),))

    for k, p in model.img_enc.named_parameters():
        if p.grad is None:
            continue
        print('{:35s}: {:8.5f}'.format(k, p.grad.data.norm(2).item(),))

    print('\n\n')
