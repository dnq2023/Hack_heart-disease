from abc import ABC, abstractmethod


class Model(ABC):
    """
    Deep learning models
    """
    def __init__(self) -> None:
        super().__init__()
        self.model = None


    @abstractmethod
    def train(self, train_data, train_label, test_data, test_label):
        pass