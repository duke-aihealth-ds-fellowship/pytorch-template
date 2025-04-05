import torch
import torch.distributions as dist
from tensordict import TensorDict
from torch import Tensor

from template.config import Config
from template.dataset import train_val_test_split


def make_parameters(
    d_features: int | list[float], scale: float, std: float, n_draws: int = 1
) -> Tensor:
    match d_features:
        case int():
            mean = torch.randn(d_features) * scale
        case list():
            mean = torch.tensor(d_features) * scale
    covariance = torch.eye(mean.size(0)) * (std**2)
    mvn = dist.MultivariateNormal(mean, covariance)
    if n_draws > 0:
        return mvn.sample((n_draws,))
    else:
        return mean


def make_latent_features(parameters: Tensor, n_samples: int, std: float) -> Tensor:
    zero = torch.zeros(parameters.size(-1))
    covariance = torch.eye(parameters.size(-1)) * std**2
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((n_samples,))
    return features


def make_observed_features(latent: Tensor, m_timepoints: int, std: float) -> Tensor:
    covariance = torch.eye(latent.shape[1]) * std**2
    mvn_ij = dist.MultivariateNormal(latent, covariance)
    features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
    return features


def make_linear_output(
    features: Tensor, parameters: Tensor, intercept: float
) -> Tensor:
    return intercept + features @ parameters.transpose(0, 1)


def make_linear_data(
    n_samples: int,
    m_timepoints: int,
    intercept: float,
    d_features: int | list[float],
    scale: float,
    parameter_std: float,
    latent_std: float,
    observed_std: float,
) -> tuple[Tensor, ...]:
    parameters = make_parameters(d_features=d_features, scale=scale, std=parameter_std)
    latent = make_latent_features(
        parameters=parameters, n_samples=n_samples, std=latent_std
    )
    observed = make_observed_features(
        latent=latent, m_timepoints=m_timepoints, std=observed_std
    )
    outputs = make_linear_output(observed, parameters, intercept)
    return parameters, latent, observed, outputs


def make_base_tensordict(
    n_samples: int,
    m_timepoints: int,
    latent: Tensor,
    observed: Tensor,
    parameters: Tensor,
) -> TensorDict:
    ids = torch.arange(n_samples)
    return TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "latent_features": latent,
            "observed_features": observed,
            "parameters": parameters.unsqueeze(0).expand(n_samples, m_timepoints, -1),
        },
        batch_size=n_samples,
    )


def regression(
    n_samples: int,
    m_timepoints: int,
    intercept: float,
    d_features: int,
    scale: float,
    parameter_std: float,
    latent_std: float,
    observed_std,
) -> TensorDict:
    parameters, latent, observed, target = make_linear_data(
        n_samples=n_samples,
        m_timepoints=m_timepoints,
        intercept=intercept,
        d_features=d_features,
        scale=scale,
        parameter_std=parameter_std,
        latent_std=latent_std,
        observed_std=observed_std,
    )
    data = make_base_tensordict(n_samples, m_timepoints, latent, observed, parameters)
    data["target"] = target.unsqueeze(1).expand(-1, m_timepoints)
    return data


def classification(
    n_samples: int,
    m_timepoints: int,
    intercept: float,
    d_features: int,
    scale: float,
    parameter_std: float,
    latent_std: float,
    observed_std: float,
) -> TensorDict:
    parameters, latent, observed, logits = make_linear_data(
        n_samples=n_samples,
        m_timepoints=m_timepoints,
        intercept=intercept,
        d_features=d_features,
        scale=scale,
        parameter_std=parameter_std,
        latent_std=latent_std,
        observed_std=observed_std,
    )
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    data = make_base_tensordict(n_samples, m_timepoints, latent, observed, parameters)
    data["probability"] = probability.unsqueeze(1).expand(-1, m_timepoints)
    data["label"] = label.unsqueeze(1).expand(-1, m_timepoints)
    return data


def time_to_event(
    n_samples: int,
    m_timepoints: int,
    intercept: float,
    d_features: int,
    scale: float,
    parameter_std: float,
    latent_std: float,
    observed_std: float,
    gamma_shape: float,
    gamma_rate: float,
) -> TensorDict:
    parameters, latent, observed, logits = make_linear_data(
        n_samples=n_samples,
        m_timepoints=m_timepoints,
        intercept=intercept,
        d_features=d_features,
        scale=scale,
        parameter_std=parameter_std,
        latent_std=latent_std,
        observed_std=observed_std,
    )
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
    time_to_event = observed_time - time
    data = make_base_tensordict(n_samples, m_timepoints, latent, observed, parameters)
    data.update(
        {
            "time": time,
            "indicator": indicator.unsqueeze(1).expand(-1, m_timepoints),
            "event_time": event_time.unsqueeze(1).expand(-1, m_timepoints),
            "censor_time": censor_time.unsqueeze(1).expand(-1, m_timepoints),
            "observed_time": observed_time.unsqueeze(1).expand(-1, m_timepoints),
            "time_to_event": time_to_event,
        }
    )
    return data


# TODO implement mixture cure model
def mixture_cure():
    pass


def simulate(cfg: Config) -> None:
    exclude = {"gamma_shape", "gamma_rate"} if cfg.task in {"tte", "mxc"} else set()
    config = cfg.simulation.model_dump(exclude=exclude)
    if cfg.task == "cls":
        data = classification(**config)
    elif cfg.task == "reg":
        data = regression(**config)
    elif cfg.task == "tte":
        data = time_to_event(**config)
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
