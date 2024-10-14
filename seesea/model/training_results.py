from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TrainingResults:
    model: str
    output_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    training_start_time: str
    training_end_time: str
    train_losses: List[float]
    val_losses: List[float]

    def to_dict(self):
        """Convert the dataclass to a dictionary"""
        return asdict(self)
