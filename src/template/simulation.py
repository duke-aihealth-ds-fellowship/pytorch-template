import torch
import torch.distributions as dist
from tensordict import TensorDict

from template.config import Config
from template.dataset import train_val_test_split


def make_parameters(
    parameters: int | list[float], scale: float, std: float, n_draws: int = 1
) -> torch.Tensor:
    match parameters:
        case int():
            mean = torch.randn(parameters) * scale
            covariance = torch.eye(parameters) * (std**2)
        case list():
            mean = torch.tensor(parameters) * scale
            covariance = torch.eye(mean.size(0)) * (std**2)
    mvn = dist.MultivariateNormal(mean, covariance)
    return mvn.sample((n_draws,))


def make_latent_features(
    d_features: int, n_samples: int, variance: float
) -> torch.Tensor:
    zero = torch.zeros(d_features)
    covariance = torch.eye(d_features) * variance
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((n_samples,))
    return features


def make_observed_features(
    features: torch.Tensor, m_timepoints: int, noise: float
) -> torch.Tensor:
    covariance = torch.eye(features.shape[1]) * noise
    mvn_ij = dist.MultivariateNormal(features, covariance)
    features = mvn_ij.sample((m_timepoints,)).permute(1, 0, 2)
    return features


def make_linear_output(
    features: torch.Tensor, parameters: torch.Tensor, intercept: float
) -> torch.Tensor:
    return intercept + features @ parameters.transpose(0, 1)


def make_linear_data(
    d_features: int,
    scale: float,
    std: float,
    n_samples: int,
    variance: float,
    m_timepoints: int,
    noise: float,
    intercept: float,
):
    parameters = make_parameters(d_features=d_features, scale=scale, std=std)
    latent_features = make_latent_features(d_features, n_samples, variance)
    observed_features = make_observed_features(latent_features, m_timepoints, noise)
    outputs = make_linear_output(latent_features, parameters, intercept)
    return parameters, observed_features, outputs


def create_base_tensor_dict(
    n_samples: int,
    m_timepoints: int,
    observed_features: torch.Tensor,
    parameters: torch.Tensor,
) -> TensorDict:
    ids = torch.arange(n_samples)
    return TensorDict(
        {
            "id": ids.unsqueeze(1).expand(-1, m_timepoints),
            "features": observed_features,
            "parameters": parameters.unsqueeze(0).expand(n_samples, m_timepoints, -1),
        },
        batch_size=n_samples,
    )


def classification(
    intercept: float,
    d_features: int,
    scale: float,
    n_samples: int,
    m_timepoints: int,
    variance: float,
    noise: float,
) -> TensorDict:
    parameters, observed_features, logits = make_linear_data(
        d_features=d_features,
        scale=scale,
        n_samples=n_samples,
        variance=variance,
        m_timepoints=m_timepoints,
        noise=noise,
        intercept=intercept,
    )
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    data = create_base_tensor_dict(
        n_samples, m_timepoints, observed_features, parameters
    )
    data["label"] = label.unsqueeze(1).expand(-1, m_timepoints)
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
    parameters, observed_features, target = make_linear_data(
        d_features=d_features,
        scale=scale,
        n_samples=n_samples,
        variance=variance,
        m_timepoints=m_timepoints,
        noise=noise,
        intercept=intercept,
    )
    data = create_base_tensor_dict(
        n_samples, m_timepoints, observed_features, parameters
    )
    data["target"] = target.unsqueeze(1).expand(-1, m_timepoints)
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
    parameters, observed_features, logits = make_linear_data(
        d_features=d_features,
        scale=scale,
        n_samples=n_samples,
        variance=variance,
        m_timepoints=m_timepoints,
        noise=noise,
        intercept=intercept,
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
    data = create_base_tensor_dict(
        n_samples, m_timepoints, observed_features, parameters
    )
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
    exclude = {"gamma_shape", "gamma_rate"}
    if cfg.task in {"tte", "mxc"}:
        exclude = set()
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
    splits.save(cfg.path.dataset)
