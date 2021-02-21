"""Mocks mainly used for testing protocols (from legacy sail-on-client)."""

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

import tinker


class MockDetector(tinker.algorithm.Algorithm):
    """Mock Detector for testing image classification protocols."""

    def __init__(self):
        """
        Detector constructor.

        Args:
            config (dict): Algorithm configuration parameters.
        """
        self.step_dict: Dict[str, Callable] = {
            "Initialize": self._initialize,
            "FeatureExtraction": self._feature_extraction,
            "WorldDetection": self._world_detection,
            "NoveltyClassification": self._novelty_classification,
            "NoveltyAdaption": self._novelty_adaption,
            "NoveltyCharacterization": self._novelty_characterization,
        }

    def get_config(self):
        """
        Implementation of smqtk Algorithm abstract method.
        """
        return {}

    def execute(self, step_descriptor: str, *args, **kwargs):
        """
        Execute method used by the protocol to run different steps associated
        with the algorithm.

        Args:
            step_descriptor (str): Name of the step.
        """
        logging.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](*args, **kwargs)

    def _initialize(self, config: Dict):
        self.config = config

    def _feature_extraction(self, fpaths: List[str]):
        """
        Feature extraction step for the algorithm.

        Args:
            fpaths (List[str]): A list of input image filepaths.

        Returns:
            tuple: (features_dict, logits_dict)
                - features_dict (dict): A dict containing a feature vector for
                  each input image in `fpaths`.
                - logits_dict (dict): A dict of logits for each input image.
        """
        features_dict = {}
        logits_dict = {}

        num_classes = 80
        num_features = random.randint(10, 100)
        for fpath in fpaths:
            logging.info(f"Extracting features for {fpath}")
            features[fpath] = np.random.randn(num_features)
            logits[fpath] = np.random.randn(num_classes)
        return features, logits

    def _world_detection(self, features: dict, red_light: str = ""):
        """
        Detect change in world (Novelty has been introduced).

        Args:
            features (dict):

        Return:
            path to csv file containing the results for change in world
        """
        return ""

    def _novelty_classification(self, toolset: str) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return ""

    def _novelty_adaption(self, feedback):
        """
        Update models based on novelty classification and characterization.

        Return:
            None
        """
        pass

    def _novelty_characterization(self, features: dict):
        """
        Characterize novelty by clustering different novel samples.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return ""
