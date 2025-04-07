import torch
import torch.distributions as dist
from tensordict import TensorDict
from torch import Tensor

from template.config import Config, SimulationConfig
from template.dataset import train_val_test_split


def make_parameters(cfg: SimulationConfig) -> Tensor:
    match cfg.d_features:
        case int():
            mean = torch.randn(cfg.d_features) * cfg.scale
        case list():
            mean = torch.tensor(cfg.d_features) * cfg.scale
    covariance = torch.eye(mean.size(0)) * (cfg.parameter_std**2)
    mvn = dist.MultivariateNormal(mean, covariance)
    if cfg.n_draws > 0:
        return mvn.sample((cfg.n_draws,))
    else:
        return mean.unsqueeze(0)


def make_latent_features(parameters: Tensor, cfg: SimulationConfig) -> Tensor:
    zero = torch.zeros(parameters.size(-1))
    covariance = torch.eye(parameters.size(-1)) * cfg.latent_std**2
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((cfg.n_samples,))
    return features


def make_observed_features(latent: Tensor, cfg: SimulationConfig) -> Tensor:
    covariance = torch.eye(latent.shape[1]) * cfg.observed_std**2
    mvn_ij = dist.MultivariateNormal(latent, covariance)
    features = mvn_ij.sample((cfg.m_timepoints,)).permute(1, 0, 2)
    return features


def make_linear_output(
    features: Tensor, parameters: Tensor, cfg: SimulationConfig
) -> Tensor:
    return cfg.intercept + features @ parameters.transpose(0, 1)


def make_linear_data(cfg: SimulationConfig) -> tuple[Tensor, ...]:
    parameters = make_parameters(cfg=cfg)
    latent = make_latent_features(parameters=parameters, cfg=cfg)
    observed = make_observed_features(latent=latent, cfg=cfg)
    outputs = make_linear_output(observed, parameters, cfg)
    return parameters, latent, observed, outputs


def make_base_tensordict(
    latent: Tensor,
    observed: Tensor,
    parameters: Tensor,
    cfg: SimulationConfig,
) -> TensorDict:
    ids = torch.arange(cfg.n_samples)
    return TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, cfg.m_timepoints),
            "latent_features": latent,
            "observed_features": observed,
            "parameters": parameters.unsqueeze(0).expand(
                cfg.n_samples, cfg.m_timepoints, -1
            ),
        },
        batch_size=cfg.n_samples,
    )


def regression(cfg: SimulationConfig) -> TensorDict:
    parameters, latent, observed, target = make_linear_data(cfg=cfg)
    data = make_base_tensordict(latent, observed, parameters, cfg)
    data["target"] = target.unsqueeze(1).expand(-1, cfg.m_timepoints)
    return data


def classification(cfg: SimulationConfig) -> TensorDict:
    parameters, latent, observed, logits = make_linear_data(cfg=cfg)
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    data = make_base_tensordict(latent, observed, parameters, cfg)
    data["probability"] = probability.unsqueeze(1).expand(-1, cfg.m_timepoints)
    data["label"] = label.unsqueeze(1).expand(-1, cfg.m_timepoints)
    return data


def time_to_event(cfg: SimulationConfig) -> TensorDict:
    parameters, latent, observed, logits = make_linear_data(cfg=cfg)
    event_rate = torch.exp(logits)
    event_time = dist.Exponential(event_rate).sample()
    gamma = dist.Gamma(cfg.gamma_shape, cfg.gamma_rate)
    time_intervals = gamma.sample((cfg.n_samples, cfg.m_timepoints))
    time = time_intervals.cumsum(dim=1)
    min_time = torch.amin(time, dim=1)
    max_time = torch.amax(time, dim=1)
    censor_time = dist.Uniform(min_time, max_time).sample()
    indicator = (event_time < censor_time).float()
    observed_time = torch.minimum(censor_time, event_time)
    time_to_event = observed_time - time
    data = make_base_tensordict(latent, observed, parameters, cfg)
    data.update(
        {
            "time": time,
            "indicator": indicator.unsqueeze(1).expand(-1, cfg.m_timepoints),
            "event_time": event_time.unsqueeze(1).expand(-1, cfg.m_timepoints),
            "censor_time": censor_time.unsqueeze(1).expand(-1, cfg.m_timepoints),
            "observed_time": observed_time.unsqueeze(1).expand(-1, cfg.m_timepoints),
            "time_to_event": time_to_event,
        }
    )
    return data


# TODO implement mixture cure model
def mixture_cure():
    pass


def simulate(cfg: Config) -> None:
    if cfg.task == "cls":
        data = classification(cfg=cfg.simulation)
    elif cfg.task == "reg":
        data = regression(cfg=cfg.simulation)
    elif cfg.task == "tte":
        data = time_to_event(cfg=cfg.simulation)
    elif cfg.task == "mxc":
        # TODO implement mixture cure model
        raise NotImplementedError("Mixture cure model is not implemented yet.")
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Choose from 'cls', 'reg', 'tte', 'mxc'."
        )
    if indicator := data.get("indicator"):
        print("Prevalence:", indicator.mean().item())
    splits = train_val_test_split(data=data, proportions=cfg.proportions)
    splits.save(str(cfg.path.dataset))
