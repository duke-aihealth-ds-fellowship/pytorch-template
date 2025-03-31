import torch
import torch.distributions as dist
from tensordict import TensorDict

from template.config import Config


def classification(
    intercept: float,
    d_features: int,
    scale: float,
    n_samples: int,
    m_timepoints: int,
    variance: float,
    noise: float,
) -> TensorDict:
    parameters = torch.randn(d_features) * scale
    zero = torch.zeros(parameters.size(0))
    identity = torch.eye(parameters.size(0))
    covariance = identity * variance
    mvn_i = dist.MultivariateNormal(zero, covariance)
    latent_features = mvn_i.sample((n_samples,))
    logits = intercept + latent_features @ parameters
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    mvn_ij = dist.MultivariateNormal(latent_features, identity * noise)
    observed_features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
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
    parameters = torch.randn(d_features) * scale
    zero = torch.zeros(parameters.size(0))
    identity = torch.eye(parameters.size(0))
    covariance = identity * variance
    mvn_i = dist.MultivariateNormal(zero, covariance)
    latent_features = mvn_i.sample((n_samples,))
    output = intercept + latent_features @ parameters
    mvn_ij = dist.MultivariateNormal(latent_features, identity * noise)
    observed_features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
    ids = torch.arange(n_samples)
    data = TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "features": observed_features,
            "target": output.unsqueeze(1).expand(-1, m_timepoints),
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
    parameters = torch.randn(d_features) * scale
    zero = torch.zeros(parameters.size(0))
    identity = torch.eye(parameters.size(0))
    sigma = identity * variance
    mvn_i = dist.MultivariateNormal(zero, sigma)
    x = mvn_i.sample((n_samples,))
    logits = intercept + x @ parameters
    event_rate = torch.exp(logits)
    event_time = dist.Exponential(event_rate).sample()
    gamma = dist.Gamma(gamma_shape, gamma_rate)
    time_intervals = gamma.sample((n_samples, m_timepoints))
    time = time_intervals.cumsum(dim=1)
    mvn_ij = dist.MultivariateNormal(x, identity * noise)
    features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
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
            "features": features,
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
