from typing import List, Tuple

from .ond_adapter import ONDAdapter
from .random_novelty_detector import RandomNoveltyDetector

class RandomNoveltyDetectorAdapter(ONDAdapter):
    def __init__(self):
        self.detector = RandomNoveltyDetector()

    def get_config(self):
        """
        Implementation of smqtk Algorithm abstract method.
        """
        return {}

    def initialize(self, config_params: dict) -> None:
        self.detector.initialize(config_params)

    def feature_extraction(self, test_params: dict) -> Tuple[dict, dict]:
        fpaths = test_params["dataset_ids"]
        features_dict, logits_dict = self.detector.feature_extraction(fpaths)
        return features_dict, logits_dict

    def world_detection(self, test_params: dict, test_data: dict) -> str:
        result = self.detector.world_detection(
            test_data["features_dict"], test_data["logits_dict"],
            red_light_image=test_params["red_light_image"],
            round_id=test_data["round_id"]
        )
        return result

    def novelty_adaption(self, test_params: dict, test_data: dict):
        return self.detector.novelty_adaption(test_data["features_dict"])

    def novelty_classification(self, test_params: dict, test_data: dict):
        result = self.detector.novelty_classification(
            test_data["features_dict"], test_data["logits_dict"],
            round_id=test_data["round_id"]
        )
        return result

    def novelty_characterization(self, test_params: dict, test_data: dict):
        result = self.detector.novelty_characterization(test_data["round_id"])
        return result

