from abc import ABC, abstractmethod
from typing import List, Callable
from .separator_model import SeparatorModel

class DecompositionStrategy(ABC):
    @abstractmethod
    def choose_separators(self, separators: List[SeparatorModel]) -> List[SeparatorModel]:
        pass

    @abstractmethod
    def is_splitting_further(self, separator: SeparatorModel, level: int) -> bool:
        pass

    @abstractmethod
    def get_separator_selection_strategy(self) -> Callable[[List[SeparatorModel]], List[SeparatorModel]]:
        pass
