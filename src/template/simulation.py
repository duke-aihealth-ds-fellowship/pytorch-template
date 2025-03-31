import torch
import torch.distributions as dist
from tensordict import TensorDict

from template.config import Config


def make_parameters(d_features: int, scale: float) -> torch.Tensor:
    return torch.randn(d_features) * scale


def make_latent_features(
    d_features: int, n_samples: int, variance: float
) -> torch.Tensor:
    zero = torch.zeros(d_features)
    covariance = torch.eye(d_features) * variance
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((n_samples,))
    return features  # (n_samples, d_features)


def make_observed_features(
    features: torch.Tensor, m_timepoints: int, noise: float
) -> torch.Tensor:
    covariance = torch.eye(features.shape[1]) * noise
    mvn_ij = dist.MultivariateNormal(features, covariance)
    features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
    return features  # (n_samples, m_timepoints, d_features)


def make_linear_output(
    features: torch.Tensor, parameters: torch.Tensor, intercept: float
) -> torch.Tensor:
    return intercept + features @ parameters


def classification(
    intercept: float,
    d_features: int,
    scale: float,
    n_samples: int,
    m_timepoints: int,
    variance: float,
    noise: float,
) -> TensorDict:
    parameters = make_parameters(d_features, scale)
    latent_features = make_latent_features(d_features, n_samples, variance)
    observed_features = make_observed_features(latent_features, m_timepoints, noise)
    logits = make_linear_output(latent_features, parameters, intercept)
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    ids = torch.arange(n_samples)
    data = TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "features": observed_features,
            "label": label.unsqueeze(1).expand(-1, m_timepoints),
            "parameters": parameters,
        },
        batch_size=n_samples,
    )
    return data


def regression(
    intercept: float,
    d_features: int,
    scale: float,
    n_samples: int,
    m_timepoints: int,
    variance: float,
    noise: float,
) -> TensorDict:
    parameters = make_parameters(d_features, scale)
    latent_features = make_latent_features(d_features, n_samples, variance)
    observed_features = make_observed_features(latent_features, m_timepoints, noise)
    target = make_linear_output(latent_features, parameters, intercept)
    ids = torch.arange(n_samples)
    data = TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "features": observed_features,
            "target": target.unsqueeze(1).expand(-1, m_timepoints),
            "parameters": parameters,
        },
        batch_size=n_samples,
    )
    return data


def time_to_event(
    intercept: float,
    d_features: int,
    scale: float,
    n_samples: int,
    m_timepoints: int,
    variance: float,
    noise: float,
    gamma_shape: float,
    gamma_rate: float,
) -> TensorDict:
    parameters = make_parameters(d_features, scale)
    latent_features = make_latent_features(d_features, n_samples, variance)
    observed_features = make_observed_features(latent_features, m_timepoints, noise)
    logits = make_linear_output(latent_features, parameters, intercept)
    event_rate = torch.exp(logits)
    event_time = dist.Exponential(event_rate).sample()
    gamma = dist.Gamma(gamma_shape, gamma_rate)
    time_intervals = gamma.sample((n_samples, m_timepoints))
    time = time_intervals.cumsum(dim=1)
    min_time = torch.amin(time, dim=1)
    max_time = torch.amax(time, dim=1)
    censor_time = dist.Uniform(min_time, max_time).sample()
    indicator = (event_time < censor_time).float()
    observed_time = torch.minimum(censor_time, event_time)
    ids = torch.arange(n_samples)
    data = TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "time": time,
            "features": observed_features,
            "indicator": indicator.unsqueeze(1).expand(-1, m_timepoints),
            "event_time": event_time.unsqueeze(1).expand(-1, m_timepoints),
            "censor_time": censor_time.unsqueeze(1).expand(-1, m_timepoints),
            "observed_time": observed_time.unsqueeze(1).expand(-1, m_timepoints),
            "parameters": parameters.unsqueeze(0).expand(n_samples, -1),
        },
        batch_size=n_samples,
    )
    return data


# TODO implement mixture cure model
def mixture_cure():
    pass


def simulate(cfg: Config):
    if cfg.task == "cls":
        exclude = {"gamma_shape", "gamma_rate"}
        cls_config = cfg.simulation.model_dump(exclude=exclude)
        data = classification(**cls_config)
    elif cfg.task == "reg":
        exclude = {"gamma_shape", "gamma_rate"}
        reg_config = cfg.simulation.model_dump(exclude=exclude)
        data = regression(**reg_config)
    elif cfg.task == "tte":
        data = time_to_event(**cfg.simulation.model_dump())
    elif cfg.task == "mxc":
        # TODO implement mixture cure model
        raise NotImplementedError("Mixture cure model is not implemented yet.")
    else:
        raise ValueError(f"Unknown task: {cfg.task}. Choose from 'cls', 'tte', 'mxc'.")
    print("Prevalence:", data["indicator"].mean().item())
    return data
