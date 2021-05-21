"""Mocks mainly used for testing protocols (from legacy sail-on-client)."""

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from smqtk_core import Configurable, Pluggable


class RandomNoveltyDetector(Configurable, Pluggable):
    def __init__(self):
        """
        Detector constructor.

        Args:
            config (dict): Algorithm configuration parameters.
        """
        self.red_light_ind = False

    def get_config(self):
        """
        Implementation of smqtk Algorithm abstract method.
        """
        return {}

    def initialize(self, config_params: dict):
        self.config_params = config_params

    def feature_extraction(
        self, test_params: dict, test_data: dict
    ) -> Tuple[dict, dict]:
        """
        Feature extraction step for the algorithm.

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
        with open(test_params["dataset"]) as dataset:
            for fpath in dataset:
                logging.info(f"Extracting features for {fpath}")
                features_dict[fpath] = np.random.randn(num_features)
                logits_dict[fpath] = np.random.randn(num_classes)
        return features_dict, logits_dict

    def world_detection(self, test_params: dict, test_data: dict) -> str:
        """
        World detection on image features.

        Returns:
            path to csv file containing the results for change in world
        """

        round_id = test_data["round_id"]
        red_light_image = test_data["red_light_image"]
        features_dict = test_data["features_dict"]

        dst_fpath = f"world_detection_{round_id}.csv"

        if red_light_image in features_dict:
            self.red_light_ind = True

        prediction = 1 if self.red_light_ind else 0

        with open(dst_fpath, "w") as f:
            for image_id in features_dict:
                f.write(f"{image_id},{prediction}\n")
        logging.info(f"Writing world detection results to {dst_fpath}")

        return dst_fpath

    def novelty_classification(
        self, test_params: dict, test_data: dict
    ) -> str:
        """
        Novelty classification on image features.

        Returns:
            path to csv file containing the novelty classification results
        """
        round_id = test_data["round_id"]
        logits_dict = test_data["logits_dict"]

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
        logging.info(f"Writing novelty classification results to {dst_fpath}")

        return dst_fpath

    def novelty_adaption(self, test_params: dict, test_data: dict):
        """
        Update models based on novelty classification and characterization.

        Return:
            None
        """
        pass

    def novelty_characterization(
        self, test_params: dict, test_data: dict
    ) -> str:
        """
        Characterize novelty by clustering different novel samples.

        Args:

        Returns:
            path to csv file containing the results for novelty characterization step
        """
        round_id = test_data["round_id"]
        dst_fpath = f"novelty_characterization_{round_id}.csv"
        return dst_fpath
