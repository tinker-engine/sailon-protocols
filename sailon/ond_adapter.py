import abc
from typing import Tuple

from smqtk_core import Configurable, Pluggable


class ONDAdapter(Configurable, Pluggable, abc.ABC):
    """
    Abstract interface class for algorithm/algorithm adapters to be run with
    the OND protocol.
    """

    @abc.abstractmethod
    def get_config(self):
        """
        :meth:`get_config` method required by SMQTK.
        """

    @abc.abstractmethod
    def initialize(self, config_params: dict) -> None:
        """
        Algorithm initialization step.

        Args:
            config_params: Dict containing configuration parameters for
                algorithm initialization.
        """

    @abc.abstractmethod
    def feature_extraction(self, test_params: dict) -> Tuple[dict, dict]:
        """
        Algorithm feature extraction step.

        Args:
            test_params: Dict containing test parameters (e.g., input
                filepaths) for the algorithm feature extraction step.

        Returns: ``features_dict`` and ``logits_dict`` output by the algorithm.
        """

    @abc.abstractmethod
    def world_detection(self, test_params: dict, test_data: dict) -> str:
        """
        Algorithm world detection step.

        Args:
            test_params: Dict containing test parameters (e.g., input
                filepaths) for the algorithm feature extraction step.
            test_data: Dict containing algorithm input/output data (eg,
                features dict, logits dict).

        Returns: Path to file containing results for world detection.
        """

    @abc.abstractmethod
    def novelty_adaption(self, test_params: dict, test_data: dict) -> None:
        """
        Algorithm novelty adaption step.

        Args:
            test_params: Dict containing test parameters (e.g., input
                filepaths) for the algorithm feature extraction step.
            test_data: Dict containing algorithm input/output data (eg,
                features dict, logits dict).
        """

    @abc.abstractmethod
    def novelty_characterization(self, test_params: dict, test_data: dict) -> str:
        """
        Algorithm novelty characterization step.

        Args:
            test_params: Dict containing test parameters (e.g., input
                filepaths) for the algorithm feature extraction step.
            test_data: Dict containing algorithm input/output data (eg,
                features dict, logits dict).

        Returns: Path to file containing results for novelty characterization.
        """

    @abc.abstractmethod
    def novelty_classification(self, test_params: dict, test_data: dict) -> str:
        """
        Algorithm novelty classification step.

        Args:
            test_params: Dict containing test parameters (e.g., input
                filepaths) for the algorithm feature extraction step.
            test_data: Dict containing algorithm input/output data (eg,
                features dict, logits dict).

        Returns: Path to file containing results for novelty classification.
        """
