"""Mocks mainly used for testing protocols (from legacy sail-on-client)."""

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from smqtk_core import Configurable, Pluggable

logger = logging.getLogger(__name__)


class RandomNoveltyDetector(Configurable, Pluggable):
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
        self.red_light_ind = False

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
        logger.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](*args, **kwargs)

    def _initialize(self, config: dict):
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

        num_classes = 413
        num_features = random.randint(10, 100)
        for fpath in fpaths:
            logger.info(f"Extracting features for {fpath}")
            features_dict[fpath] = np.random.randn(num_features)
            logits_dict[fpath] = np.random.randn(num_classes)
        return features_dict, logits_dict

    def _world_detection(
        self, features_dict: dict, logits_dict: dict,
        red_light_image: str = "", round_id: int = None
    ) -> str:
        """
        World detection on image features.

        Args:
            features_dict (dict): Dict returned by :meth:`_feature_extraction`
                where each key corresponds to an image and each value to its
                respective features.
            logits_dict (dict): Dict returned by :meth:`_feature_extraction`
                where each key corresponds to an image and each value to its
                respective logits.
            red_light_image (str): TODO

        Returns:
            path to csv file containing the results for change in world
        """
        # TODO: Does `logits_dict` need to be an argument to this method?

        dst_fpath = f"world_detection_{round_id}.csv"

        if red_light_image in features_dict:
            self.red_light_ind = True

        prediction = 1 if self.red_light_ind else 0

        with open(dst_fpath, "w") as f:
            for image_id in features_dict:
                f.write(f"{image_id},{prediction}\n")
        logger.info(f"Writing world detection results to {dst_fpath}")

        return dst_fpath

    def _novelty_classification(
        self, features_dict: dict, logits_dict: dict, round_id: int = None
    ) -> str:
        """
        Novelty classification on image features.

        Args:
            features_dict (dict): Dict returned by :meth:`_feature_extraction`
                where each key corresponds to an image and each value to its
                respective features.
            logits_dict (dict): Dict returned by :meth:`_feature_extraction`
                where each key corresponds to an image and each value to its
                respective logits.
        Returns:
            path to csv file containing the novelty classification results
        """
        dst_fpath = f"novelty_classification_{round_id}.csv"

        if self.red_light_ind:
            novelty_predictions = np.ones((len(logits_dict), 1))
        else:
            novelty_predictions = np.zeros((len(logits_dict), 1))

        logits_arr = np.array(tuple(logits_dict.values()))
        softmax_scores = np.divide(
            np.exp(logits_arr),
            np.sum(np.exp(logits_arr), axis=1)[:, np.newaxis]
        )

        predictions = np.hstack((novelty_predictions, softmax_scores))

        df = pd.DataFrame(zip(logits_dict.keys(), *predictions.T))
        df.to_csv(dst_fpath, index=False, header=False, float_format="%.4f")
        logger.info(f"Writing novelty classification results to {dst_fpath}")

        return dst_fpath

    def _novelty_adaption(self, features_dict: dict):
        """
        Update models based on novelty classification and characterization.

        Return:
            None
        """
        pass

    def _novelty_characterization(
        self, features_dict: dict, logits_dict: dict, round_id: int = None
    ):
        """
        Characterize novelty by clustering different novel samples.

        Args:

        Returns:
            path to csv file containing the results for novelty characterization step
        """
        dst_fpath = f"novelty_characterization_{round_id}.csv"
        return dst_fpath
