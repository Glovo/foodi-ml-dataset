from ..utils.logger import get_logger
from .loaders import prepare_ml_data

logger = get_logger()


class DataIterator:
    def __init__(self, loader, device, non_stop=False):
        self.data_iter = iter(loader)
        self.loader = loader
        self.non_stop = non_stop
        self.device = device

    def __str__(self):
        return f"{self.loader.dataset.data_name}.{self.loader.dataset.data_split}"

    def next(self):
        try:
            instance = next(self.data_iter)

            targ_a, lens_a, targ_b, lens_b, ids = prepare_ml_data(instance, self.device)
            logger.debug(
                (
                    f"DataIter - CrossLang - Images: {targ_a.shape} "
                    f"DataIter - CrossLang - Target: {targ_a.shape} "
                    f"DataIter - CrossLang - Ids: {ids[:10]}\n"
                )
            )
            return targ_a, lens_a, targ_b, lens_b, ids

        except StopIteration:
            if self.non_stop:
                self.data_iter = iter(self.loader)
                return self.next()
            else:
                raise StopIteration("The data iterator has finished its job.")
