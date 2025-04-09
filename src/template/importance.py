from captum.attr import GradientShap
from torch import Tensor
from torch.utils.data import DataLoader

from template.config import Config
from template.model import Transformer
from template.tune import load_checkpoint


# TODO: replace embeddings
def make_attributions(inputs: Tensor, baselines: Tensor, target: int, cfg: Config):
    model = load_checkpoint(cfg=cfg, model_class=Transformer)
    inputs = inputs.to(cfg.trainer.device)
    baselines = baselines.to(cfg.trainer.device)
    model.to(cfg.trainer.device)
    lig = GradientShap(model)
    attributions, delta = lig.attribute(
        inputs=inputs,
        baselines=baselines,
        return_convergence_delta=True,
        target=target,
    )
    print("Mean convergence delta:", delta.mean().item())
    # sum attributions across embedding dimension
    attributions = attributions.sum(dim=-1)
    return attributions


# TODO:
def feature_importance(cfg: Config, loader: DataLoader):
    pass
