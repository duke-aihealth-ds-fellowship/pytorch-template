import torch
import torch.distributions as dist
from tensordict import TensorDict
from torch import Tensor

from template.config import Config, SimulationConfig
from template.dataset import train_val_test_split


def make_parameters(cfg: SimulationConfig) -> Tensor:
    match cfg.parameters:
        case int():
            mean = torch.randn(cfg.parameters)
        case list():
            mean = torch.tensor(cfg.parameters)
    mean = mean * cfg.scale
    covariance = torch.eye(mean.size(0)) * (cfg.parameter_std**2)
    mvn = dist.MultivariateNormal(mean, covariance)
    if cfg.num_draws > 0:
        return mvn.sample((cfg.num_draws,))
    else:
        return mean.unsqueeze(0)


def make_latent_features(parameters: Tensor, cfg: SimulationConfig) -> Tensor:
    zero = torch.zeros(parameters.size(-1))
    covariance = torch.eye(parameters.size(-1)) * cfg.latent_std**2
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((cfg.num_samples,))
    return features


def make_observed_features(latent: Tensor, cfg: SimulationConfig) -> Tensor:
    covariance = torch.eye(latent.shape[1]) * cfg.observed_std**2
    mvn_ij = dist.MultivariateNormal(latent, covariance)
    features = mvn_ij.sample((cfg.num_timepoints,)).permute(1, 0, 2)
    return features


def make_linear_output(
    features: Tensor, parameters: Tensor, cfg: SimulationConfig
) -> Tensor:
    output = cfg.intercept + features @ parameters.transpose(0, 1)
    return output


def make_linear_data(cfg: SimulationConfig) -> tuple[Tensor, ...]:
    parameters = make_parameters(cfg=cfg)
    latent = make_latent_features(parameters=parameters, cfg=cfg)
    observed = make_observed_features(latent=latent, cfg=cfg)
    outputs = make_linear_output(observed, parameters, cfg)
    return latent, observed, outputs


def make_base_tensordict(
    latent: Tensor, observed: Tensor, cfg: SimulationConfig
) -> TensorDict:
    ids = torch.arange(cfg.num_samples).unsqueeze(1).expand(-1, cfg.num_timepoints)
    return TensorDict(
        {"id": ids, "latent_features": latent, "observed_features": observed},
        batch_size=cfg.num_samples,
    )


def make_target_constant(target: Tensor, cfg: SimulationConfig) -> Tensor:
    target = torch.mode(target, dim=1).values
    target = target.unsqueeze(1).expand(-1, cfg.num_timepoints, -1)
    return target


def regression(cfg: SimulationConfig) -> TensorDict:
    latent, observed, target = make_linear_data(cfg=cfg)
    target = make_target_constant(target=target, cfg=cfg)
    data = make_base_tensordict(latent, observed, cfg)
    data["target"] = target
    return data


def classification(cfg: SimulationConfig) -> TensorDict:
    latent, observed, logits = make_linear_data(cfg=cfg)
    probability = torch.sigmoid(logits)
    label = torch.bernoulli(probability)
    target = make_target_constant(target=label, cfg=cfg)
    data = make_base_tensordict(latent, observed, cfg)
    data["probability"] = probability
    data["label"] = target
    return data


def time_to_event(cfg: SimulationConfig) -> TensorDict:
    latent, observed, logits = make_linear_data(cfg=cfg)
    event_rate = torch.exp(logits)
    event_time = dist.Exponential(event_rate).sample()
    gamma = dist.Gamma(cfg.gamma_shape, cfg.gamma_rate)
    time_intervals = gamma.sample((cfg.num_samples, cfg.num_timepoints))
    time = time_intervals.cumsum(dim=1).unsqueeze(-1)
    min_time = torch.amin(time, dim=1)
    max_time = torch.amax(time, dim=1)
    censor_time = dist.Uniform(min_time, max_time).sample().unsqueeze(-1)
    indicator = (event_time < censor_time).float()
    indicator = make_target_constant(target=indicator, cfg=cfg)
    observed_time = torch.minimum(censor_time, event_time)
    time_to_event = observed_time - time
    data = make_base_tensordict(latent, observed, cfg)
    data.update(
        {
            "time": time,
            "indicator": indicator,
            "event_time": event_time,
            "censor_time": censor_time,
            "observed_time": observed_time,
            "time_to_event": time_to_event,
        }
    )
    return data


# TODO implement mixture cure model
def mixture_cure():
    pass


def print_simulation_shapes(data: TensorDict) -> None:
    max_key_length = max(list(len(key) for key, _ in data.items()))
    for key, value in data.items():
        print(f"{key:<{max_key_length}} : {value.shape}")


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
    splits = train_val_test_split(data=data, proportions=cfg.proportions)
    splits.save(str(cfg.path.dataset))
    print_simulation_shapes(data=data)
    if "indicator" in data.keys():
        print("Prevalence:", data["indicator"].mean().item())
