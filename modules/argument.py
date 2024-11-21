from dataclasses import dataclass

@dataclass
class TrainArguments:
    dataset: str = 'Flickr'
    lr: float = 5e-4
    alpha: float = 0.3
    gamma: float = 0.4

    n_hidden: int = 128
    k: int = 2
    resultdir: str = 'results'
    device: str = 'cuda:0'
    seed: int = 1
    num_epoch: int = 1500
    weight_decay: float = 0.0

    batch_size: int = -1