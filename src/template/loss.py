import torch
import torch.nn as nn
from torch import Tensor


class DiscreteFailureTimeNLL(nn.Module):
    """
    Negative log-likelihood loss for discrete-time survival analysis.
    """

    def __init__(
        self, boundaries: Tensor, ignore_index: int, epsilon: float = 1e-8
    ) -> None:
        """
        Args:
            boundaries: Tensor defining the boundaries of time intervals
            epsilon: Small constant added for numerical stability
            ignore_index: Index to ignore in the loss computation
        """
        super().__init__()
        self.boundaries = nn.Buffer(boundaries.view(1, 1, -1))
        self.interval_start = nn.Buffer(boundaries[..., :-1])
        self.interval_end = nn.Buffer(boundaries[..., 1:])
        self.interval_width = nn.Buffer(self.interval_end - self.interval_start)
        self.min_proportion = nn.Buffer(torch.tensor(0.0))
        self.max_proportion = nn.Buffer(torch.tensor(1.0))
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def _identify_event_interval(self, time: Tensor) -> Tensor:
        """Determine which interval each event/censoring time falls into."""
        after_start = time > self.interval_start
        before_or_at_end = time <= self.interval_end
        return after_start & before_or_at_end

    def _follow_up_proportion(self, time: Tensor) -> Tensor:
        """Calculate the proportion of each interval that subject was exposed to risk."""
        # Time elapsed since the start of each interval
        elapsed_time = time - self.interval_start
        # Convert elapsed time to fraction of interval width
        proportion = elapsed_time / self.interval_width.clamp(min=1.0)
        # proportion = elapsed_time / self.interval_width.clamp(min=self.epsilon)
        # Clip proportion at 1.0 (full interval follow up)
        clipped_proportion = torch.minimum(proportion, self.max_proportion)
        # Ensure non-negative follow up (minimum 0.0)
        follow_up_proportion = torch.maximum(clipped_proportion, self.min_proportion)
        return follow_up_proportion

    def forward(
        self, inputs: Tensor, indicator: Tensor, time_to_event: Tensor
    ) -> Tensor:
        # (batch_size, seq_len, num_intervals)
        probabilities = torch.softmax(inputs, dim=-1)[..., :-1]
        # Find hazard at interval where event occurred
        event_time = time_to_event.unsqueeze(-1)  # (batch_size, seq_len, 1)
        event_intervals = self._identify_event_interval(event_time)
        event_likelihood = (event_intervals * probabilities).sum(dim=-1) + self.epsilon
        # Calculate survival probability at censoring time
        follow_up_proportions = self._follow_up_proportion(event_time)
        cumulative_probability = (follow_up_proportions * probabilities).sum(dim=-1)
        survival_probability = 1 - cumulative_probability + self.epsilon
        # Compute log-likelihood based on event status
        event_log_likelihood = indicator * event_likelihood.log()
        censoring_log_likelihood = (1 - indicator) * survival_probability.log()
        log_likelihood = event_log_likelihood + censoring_log_likelihood
        mask = indicator != self.ignore_index
        return -1.0 * (log_likelihood * mask).sum() / mask.sum()
