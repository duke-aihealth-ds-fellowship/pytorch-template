import numpy as np
import polars as pl


# TODO add group id for analysis stratification
def make_fake_sequence_dataset() -> pl.DataFrame:
    """Generates random embedding indices to minic the struction sequence datasets.

    Returns:
        polars.DataFrame: A DataFrame with columns "instance_id", "input", and "label".
    """
    n_samples = 200
    max_sequence_length = 10
    n_classes = 2
    vocab_size = 50
    sequence_ids = np.arange(n_samples)
    sequence_lengths = np.random.randint(1, max_sequence_length + 1, size=n_samples)
    sequences = [
        np.random.randint(0, vocab_size, size=length) for length in sequence_lengths
    ]
    labels = np.random.randint(0, n_classes, size=n_samples).tolist()
    return pl.DataFrame(
        {"instance_id": sequence_ids, "input": sequences, "label": labels}
    )
