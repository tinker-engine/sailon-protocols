import abc
from typing import Tuple

#from smqtk_core import Plugfigurable
from smqtk_core import Configurable, Pluggable


class ONDAdapter(Configurable, Pluggable, abc.ABC):
    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self, config_params: dict) -> None:
        pass

    @abc.abstractmethod
    def feature_extraction(self, test_params: dict) -> Tuple[dict, dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def world_detection(self, test_params: dict, test_data: dict) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def novelty_adaption(self, test_params: dict, test_data: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def novelty_characterization(self, test_params: dict, test_data: dict) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def novelty_classification(self, test_params: dict, test_data: dict) -> str:
        raise NotImplementedError
