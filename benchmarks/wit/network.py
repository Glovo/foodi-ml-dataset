# define embedding heads similar to what ponc did
import torch
import torch.nn as nn
import torchvision
from sentence_transformers import SentenceTransformer, models, util


class NeuralNetwork:
    """Base class for neural network. This class contains an attribute self.network that must instantiate a torch.nn.Module object."""

    def __init__(self):
        self.network = None

    def change_trainable_parameters(self, layers_to_train, train: bool = True):
        """
        Change the trainable parameters of a network. Iterates over network parameters and change them to be trainable or not.
        Parameters
        ----------
        layers_to_train : List
        List with the name of the layers that should be trained or frozen.
        train : bool
        Whether the parameter will be updated in the next step of the optimizer.
        """
        if layers_to_train == ["all"]:
            for _, param in self.network.named_parameters():
                param.requires_grad = train
        else:
            for name, param in self.network.named_parameters():
                requires_training = train if name in layers_to_train else not train
                param.requires_grad = requires_training


class LanguageEmbeddingsHead(NeuralNetwork):
    def __init__(self, model_type="stsb-xlm-r-multilingual"):
        # NOTE: https://www.sbert.net/docs/training/overview.html
        super().__init__()
        embedding_dim_reduction = models.Dense(in_features=768, out_features=512)
        self.network = SentenceTransformer(
            modules=[SentenceTransformer(model_type), embedding_dim_reduction]
        )


class ImageEmbeddingsCNN(nn.Module, NeuralNetwork):
    """CNN for the image tower of the WIT method"""

    def __init__(self, pretrained=True):
        super(ImageEmbeddingsCNN, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=pretrained)
        linear_layer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.Tanh(),
        )
        self.network.fc = linear_layer

    def forward(self, x):
        return self.network(x)


class WIT_NN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn = ImageEmbeddingsCNN()
        self.language_head = LanguageEmbeddingsHead()
        self.device = kwargs["device"]

    def forward(self, batch):
        text_embeds = self.language_head.network.tokenize(batch["caption"])
        text_embeds = util.batch_to_device(text_embeds, self.device)
        text_embeds = self.language_head.network(text_embeds)["sentence_embedding"]
        text_embeds = self.norm(text_embeds)
        batch_img = batch["img"].to(self.device)
        imge_embeds = self.cnn(batch_img)
        imge_embeds = self.norm(imge_embeds)
        # compute similarity matrix
        sim = self.sim_matrix(text_embeds, imge_embeds)
        return sim

    @torch.no_grad()
    def forward_embeds(self, batch):
        text_embeds = self.language_head.network.tokenize(batch["caption"])
        text_embeds = util.batch_to_device(text_embeds, self.device)
        text_embeds = self.language_head.network(text_embeds)["sentence_embedding"]
        text_embeds = self.norm(text_embeds)
        batch_img = batch["img"].to(self.device)
        imge_embeds = self.cnn(batch_img)
        imge_embeds = self.norm(imge_embeds)
        return imge_embeds, text_embeds

    def norm(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        sim_mt = torch.mm(a, b.transpose(0, 1))
        return sim_mt


def load_saved_model(device, path="./trained_model.pth"):
    model = WIT_NN(device=device)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.language_head.network = model.language_head.network.to(device)
    model.cnn.network = model.cnn.network.to(device)
    return model


def model_to_device(device, model):
    model = model.to(device)
    model.language_head.network = model.language_head.network.to(device)
    model.cnn.network = model.cnn.network.to(device)
    return model
