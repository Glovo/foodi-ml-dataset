import torch
import torch.nn as nn

from ..utils.logger import get_logger
from . import data_parallel
from .imgenc import get_image_encoder, get_img_pooling
from .similarity.factory import get_similarity_object
from .similarity.similarity import Similarity, Similarity_Ev
from .txtenc import get_text_encoder, get_txt_pooling

logger = get_logger()


class Retrieval(nn.Module):
    def __init__(
        self,
        txt_enc={},
        img_enc={},
        similarity={},
        ml_similarity={},
        tokenizers=None,
        latent_size=1024,
        **kwargs,
    ):
        super().__init__()

        self.master = True
        self.latent_size = latent_size
        self.img_enc = get_image_encoder(
            name=img_enc.name, latent_size=latent_size, **img_enc.params
        )

        logger.info(("Image encoder created: " f"{img_enc.name,}"))

        self.txt_enc = get_text_encoder(
            name=txt_enc.name,
            latent_size=latent_size,
            tokenizers=tokenizers,
            **txt_enc.params,
        )

        self.tokenizers = tokenizers

        self.txt_pool = get_txt_pooling(txt_enc.pooling)
        self.img_pool = get_img_pooling(img_enc.pooling)

        logger.info(("Text encoder created: " f"{txt_enc.name}"))

        sim_obj = get_similarity_object(similarity.name, **similarity.params)
        sim_obj_eval = get_similarity_object("adapt_i2t_eval", **similarity.params)
        self.similarity = Similarity(
            similarity_object=sim_obj,
            device=similarity.device,
            latent_size=latent_size,
            **kwargs,
        )
        self.similarity_eval = Similarity_Ev(
            similarity_object=sim_obj_eval,
            device=similarity.device,
            latent_size=latent_size,
            **kwargs,
        )

        self.ml_similarity = nn.Identity()
        if ml_similarity is not None:
            self.ml_similarity = self.similarity

            if ml_similarity != {}:
                ml_sim_obj = get_similarity_object(
                    ml_similarity.name, **ml_similarity.params
                )

                self.ml_similarity = Similarity(
                    similarity_object=ml_sim_obj,
                    device=similarity.device,
                    latent_size=latent_size,
                    **kwargs,
                )

        self.set_devices_()

        logger.info(f"Using similarity: {similarity.name,}")

    def set_devices_(
        self, txt_devices=["cuda"], img_devices=["cuda"], loss_device="cuda"
    ):
        if len(txt_devices) > 1:
            self.txt_enc = data_parallel.DataParallel(self.txt_enc)
            self.txt_enc.device = torch.device("cuda")
        elif len(txt_devices) == 1:
            try:
                self.txt_enc.to(txt_devices[0])
            except:
                self.txt_enc.to(txt_devices[0])
            self.txt_enc.device = torch.device(txt_devices[0])

        if len(img_devices) > 1:
            self.img_enc = data_parallel.DataParallel(self.img_device)
            self.img_enc.device = torch.device("cuda")
        elif len(img_devices) == 1:
            self.img_enc.to(img_devices[0])
            self.img_enc.device = torch.device(img_devices[0])

        self.loss_device = torch.device(loss_device)

        self.similarity = self.similarity.to(self.loss_device)
        self.similarity_eval = self.similarity.to(self.loss_device)

        # self.ml_similarity = self.ml_similarity.to(self.loss_device)

        logger.info(
            (
                f"Setting devices: "
                f"img: {self.img_enc.device},"
                f"txt: {self.txt_enc.device}, "
                f"loss: {self.loss_device}"
            )
        )

    def embed_caption_features(self, cap_features, lengths):
        return self.txt_pool(cap_features, lengths)

    def embed_image_features(self, img_features):
        return self.img_pool(img_features)

    def embed_images(self, batch):
        img_tensor = self.img_enc(batch)
        img_embed = self.embed_image_features(img_tensor)
        return img_embed

    def embed_captions(self, batch):
        txt_tensor, lengths = self.txt_enc(batch)
        txt_embed = self.embed_caption_features(txt_tensor, lengths)
        return txt_embed

    def forward_batch(self, batch):
        img_embed = self.embed_images(batch["image"].to(self.img_enc.device))
        txt_embed = self.embed_captions(batch)
        return img_embed, txt_embed

    # def forward(self, images, captions, lengths):
    #     img_embed = self.embed_images(images)
    #     txt_embed = self.embed_captions(captions, lengths)
    #     return img_embed, txt_embed

    def get_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.similarity(embed_a, embed_b, lens)

    def get_sim_matrix_eval(self, embed_a, embed_b, lens=None, shared_size=128):
        return self.similarity.forward_shared_eval(
            embed_a, embed_b, lens, shared_size=shared_size
        )

    def get_ml_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.ml_similarity(embed_a, embed_b, lens)

    def get_sim_matrix_shared(self, embed_a, embed_b, lens=None, shared_size=128):
        return self.similarity.forward_shared(
            embed_a, embed_b, lens, shared_size=shared_size
        )
