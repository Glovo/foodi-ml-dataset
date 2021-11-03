import os
import random
from pathlib import Path
from timeit import default_timer as dt

import numpy as np
import torch
from addict import Dict
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import dataset
from tqdm import tqdm

from ..data.dataiterator import DataIterator
from ..utils import file_utils, helper, logger
from . import evaluation, optimizers
from .lr_scheduler import get_scheduler

torch.manual_seed(0)
random.seed(0, version=2)


def freeze(module):
    for x in module.parameters():
        x.requires_grad = False


class Trainer:
    def __init__(
        self,
        model=None,
        device=torch.device("cuda"),
        world_size=1,
        args=None,
        sysoutlog=tqdm.write,
        master=True,
        path="runs/",
    ):
        self.model = model
        self.device = device
        self.train_logger = logger.LogCollector()
        self.val_logger = logger.LogCollector()
        self.args = args
        self.sysoutlog = sysoutlog
        self.optimizer = None
        self.metrics = {}
        self.master = master
        self.world_size = world_size
        self.continue_training = True
        self.path = path
        self.save_every_n_epoch = 1
        self.path_checkpoints = "runs"
        self.best_model_path = Path(self.path_checkpoints) / Path(
            "best_model_foodi.pkl"
        )
        os.makedirs(self.path_checkpoints, exist_ok=True)

    def setup_optim(
        self,
        optimizer={},
        lr=1e-3,
        lr_scheduler=None,
        clip_grad=2.0,
        log_histograms=False,
        log_grad_norm=False,
        early_stop=50,
        freeze_modules=[],
        **kwargs,
    ):
        # TODO: improve this!
        count_params = lambda p: np.sum([np.product(tuple(x.shape)) for x in p])
        total_params = count_params(self.model.parameters())

        for fmod in freeze_modules:
            print(f"Freezing {fmod}")
            freeze(eval(f"self.{fmod}"))

        trainable_params = [x for x in self.model.parameters() if x.requires_grad]

        self.optimizer = optimizers.get_optimizer(
            optimizer.name,
            trainable_params,
            **optimizer.params,
        )

        scheduler = None
        if lr_scheduler.name is not None:
            scheduler = get_scheduler(
                lr_scheduler.name, self.optimizer, **lr_scheduler.params
            )

        for k in self.optimizer.param_groups:
            self.sysoutlog(
                f"lr: {k['lr']}, #layers: {len(k['params'])}, #params: {count_params(k['params']):,}"
            )

        self.sysoutlog(f"Total Params: {total_params:,}, ")
        self.initial_lr = lr
        self.lr_scheduler = scheduler
        self.clip_grad = clip_grad
        self.log_histograms = log_histograms
        self.log_grad_norm = False
        self.save_all = False
        self.best_val = 0
        self.count = early_stop
        self.early_stop = early_stop

    def fit(
        self,
        train_loader,
        valid_loaders,
        lang_loaders=[],
        init_iteration=0,
        nb_epochs=2000,
        log_interval=50,
        valid_interval=500,
        world_size=1,
    ):

        self.tb_writer = helper.set_tensorboard_logger(self.path)
        self.path = self.tb_writer.file_writer.get_logdir()
        file_utils.save_yaml_opts(Path(self.path) / "options.yaml", self.args)

        self.check_optimizer_setup()
        pbar = lambda x: range(x)
        if self.master:
            pbar = lambda x: tqdm(range(x), desc="Epochs")

        for epoch in pbar(nb_epochs):
            self.train_epoch(
                train_loader=train_loader,
                lang_loaders=lang_loaders,
                epoch=epoch,
                log_interval=log_interval,
                valid_loaders=valid_loaders,
                valid_interval=valid_interval,
                path=self.path,
            )

            # Save checkpoint of foodiml
            if (epoch % self.save_every_n_epoch) == 0:
                print(f"Saving epoch {epoch} ...")
                self.save_foodi(path=self.path_checkpoints, is_best=False, epoch=epoch)

            if not self.continue_training:
                break

        # Save the final epoch
        print(f"Saving final epoch {epoch} ...")
        self.save_foodi(path=self.path_checkpoints, is_best=True, epoch=epoch)

    def check_optimizer_setup(self):
        if self.optimizer is None:
            print("You forgot to setup_optim.")
            exit()

    def _forward_multimodal_loss(self, batch):
        img_emb, cap_emb = self.model.forward_batch(batch)
        lens = batch["caption"][1]
        sim_matrix = self.model.get_sim_matrix(img_emb, cap_emb, lens)
        # sim_matrix = self.model.get_sim_matrix_eval(img_emb, cap_emb, lens)
        loss = self.model.mm_criterion(sim_matrix)
        return loss

    def _forward_multilanguage_loss(
        self, captions_a, lens_a, captions_b, lens_b, *args
    ):
        cap_a_embed = self.model.embed_captions({"caption": (captions_a, lens_a)})
        cap_b_embed = self.model.embed_captions({"caption": (captions_b, lens_b)})

        if len(cap_a_embed.shape) == 3:
            from ..model.txtenc import pooling

            cap_a_embed = pooling.last_hidden_state_pool(cap_a_embed, lens_a)
            cap_b_embed = pooling.last_hidden_state_pool(cap_b_embed, lens_b)

        sim_matrix = self.model.get_ml_sim_matrix(cap_a_embed, cap_b_embed, lens_b)
        loss = self.model.ml_criterion(sim_matrix)
        return loss

    def _get_lang_iters(self, lang_loaders):
        lang_iters = [
            DataIterator(loader=loader, device=self.device, non_stop=True)
            for loader in lang_loaders
        ]
        return lang_iters

    def _get_multilanguage_total_loss(self, lang_iters):
        total_lang_loss = 0.0
        loss_info = {}
        for lang_iter in lang_iters:
            lang_data = lang_iter.next()
            lang_loss = self._forward_multilanguage_loss(*lang_data)
            total_lang_loss += lang_loss
            loss_info[f"train_loss_{str(lang_iter)}"] = lang_loss
        return total_lang_loss, loss_info

    def train_epoch(
        self,
        train_loader,
        lang_loaders,
        epoch,
        valid_loaders=[],
        log_interval=50,
        valid_interval=500,
        path="",
    ):
        lang_iters = self._get_lang_iters(lang_loaders)

        for batch in train_loader:
            train_info = self._forward(batch, lang_iters, epoch)
            self._update_tb_log_info(train_info)

            if self.model.mm_criterion.iteration % valid_interval == 0:
                self.run_evaluation(valid_loaders)

            if self.model.mm_criterion.iteration % log_interval == 0 and self.master:
                self._print_log_info(train_info)

    def _forward(self, batch, lang_iters, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        begin_forward = dt()

        multimodal_loss = self._forward_multimodal_loss(batch)
        total_lang_loss, loss_info = self._get_multilanguage_total_loss(lang_iters)
        total_loss = multimodal_loss + total_lang_loss
        total_loss.backward()

        norm = 0.0
        if self.clip_grad > 0:
            norm = clip_grad_norm_(self.model.parameters(), self.clip_grad)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        end_backward = dt()
        batch_time = end_backward - begin_forward
        return self._update_train_info(
            batch_time, multimodal_loss, total_loss, epoch, norm, loss_info
        )

    def _print_log_info(self, train_info):
        helper.print_tensor_dict(train_info, print_fn=self.sysoutlog)
        if self.log_histograms:
            logger.log_param_histograms(
                self.model, self.tb_writer, self.model.mm_criterion.iteration
            )

    def _update_train_info(
        self, batch_time, multimodal_loss, total_loss, epoch, norm, loss_info
    ):
        train_info = Dict(
            {
                "loss": multimodal_loss,
                "iteration": self.model.mm_criterion.iteration,
                "total_loss": total_loss,
                "k": self.model.mm_criterion.k,
                "batch_time": batch_time,
                "countdown": self.count,
                "epoch": epoch,
                "norm": norm,
            }
        )
        train_info.update(loss_info)
        for param_group in self.optimizer.param_groups:
            if "name" in param_group:
                train_info.update({f"lr_{param_group['name']}": param_group["lr"]})
            else:
                train_info.update({"lr_base": param_group["lr"]})
        return train_info

    def _update_tb_log_info(self, train_info):
        if self.master:
            logger.tb_log_dict(
                tb_writer=self.tb_writer,
                data_dict=train_info,
                iteration=self.model.mm_criterion.iteration,
                prefix="train",
            )

    def run_evaluation(self, valid_loaders):
        metrics, val_metric = self.evaluate_loaders(valid_loaders)
        self._update_early_stop_vars(val_metric)
        if self.master:
            self.save(
                path=self.path,
                is_best=(val_metric >= self.best_val),
                args=self.args,
                rsum=val_metric,
            )
            for metric, values in metrics.items():
                self.tb_writer.add_scalar(
                    metric, values, self.model.mm_criterion.iteration
                )
        self._check_early_stop()

    def _check_early_stop(self):
        if self.count == 0 and self.master:
            self.sysoutlog("\n\nEarly stop\n\n")
            self.continue_training = False

    def _update_early_stop_vars(self, val_metric):
        if val_metric < self.best_val:
            self.count -= 1
        elif not self.save_all:
            self.count = self.early_stop
            self.best_val = val_metric

    def evaluate_loaders(self, loaders):
        loader_metrics = {}
        final_sum = 0.0
        nb_loaders = len(loaders)
        print("nb_loaders: ", nb_loaders)
        for i, loader in enumerate(loaders):
            loader_name = str(loader.dataset)
            self.sysoutlog(f"Evaluating {i+1:2d}/{nb_loaders:2d} - {loader_name}")
            print("evaluation.predict_loader begins")
            img_emb, txt_emb, lens = evaluation.predict_loader_smart(
                self.model, loader, self.device
            )
            print("Beginning evaluation.evaluate")
            result = evaluation.evaluate(
                model=self.model,
                img_emb=img_emb,
                txt_emb=txt_emb,
                lengths=lens,
                device=self.device,
                shared_size=128,
            )

            for k, v in result.items():
                self.sysoutlog(f"{k:<10s}: {v:>6.1f}")

            result = {
                f"{loader_name}/{metric_name}": v for metric_name, v in result.items()
            }

            loader_metrics.update(result)
            final_sum += result[f"{loader_name}/rsum"]
        return loader_metrics, final_sum / float(nb_loaders)

    def save_foodi(self, path=None, is_best=False, epoch=0):
        helper.save_checkpoint_foodi(
            path,
            self.model,
            optimizer=self.optimizer,
            is_best=is_best,
            epoch=epoch,
        )

    def save(self, path=None, is_best=False, args=None, **kwargs):
        helper.save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            is_best=is_best,
            save_all=self.save_all,
            iteration=self.model.mm_criterion.iteration,
            args=self.args,
            **kwargs,
        )

    def load(self, path=None):
        if path is None:
            path = self.best_model_path
        states = helper.restore_checkpoint(path, self.model, None)
        self.model = states["model"].to(self.device)

    def __repr__(
        self,
    ):
        string = f"{type(self).__name__} " f"{type(self.model).__name__} "
        return string
