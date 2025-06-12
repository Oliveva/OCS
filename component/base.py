from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseComponent(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def monitor(self) -> Dict[str, Any]:
        pass
