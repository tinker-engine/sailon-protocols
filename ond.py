import itertools
import logging

#from dummy_interface import DummyInterface
#from mock import MockDetector
from sailon import DummyInterface, MockDetector

import tinker


class ONDProtocol(tinker.protocol.Protocol):
    def __init__(self):
        super().__init__()
        self.interface = DummyInterface()

    def get_config(self):
        return {
            "domain": "image_classification",
            "test_ids": [],
            "novelty_detector_class": "",
            "seed": "seed",
            "dataset_root": "",
            "feature_extraction_only": False,
            "use_feedback": False,
            "save_features": False,
            "use_saved_features": False,
            "save_attributes": False,
            "use_saved_attributes": False,
            "save_elementwise": False,
            "saved_attributes": {},
            "skip_stage": [],
            "hints": [],
            "detector_config": {
                "efficientnet_params": {
                    "model_path": "",
                    "known_classes": 413,
                },
                "evm_params": {
                    "model_path": "",
                    "tailsize": 33998,
                    "cover_threshold": 0.7,
                    "distance_multiplier": 0.55,
                },
                "known_kmeans_params": {},
                "dataloader_params": {"batch_size": 64, "num_workers": 3},
                "csv_folder": "",
                "cores": 4,
                "detection_threshold": 0.3,
            },
        }

    def run_protocol(self, config_):
        config = self.get_config()
        config.update(config_)

        # smqtk will ultimately handle retrieval of the algorithm.
        algorithm = MockDetector()

        session_id = self.interface.new_session(
            test_ids=config["test_ids"], protocol="OND",
            domain=config["domain"],
            novelty_detector_spec="1.0.0.Mock",
            hints=config["hints"]
        )

        for test_id in config["test_ids"]:
            # Assume save_attributes == False and skip for now.
            test_metadata = self.interface.get_test_metadata(
                session_id=session_id,
                test_id=test_id
            )

            test_params = {}

            red_light = test_metadata.get("red_light", "")

            # Assume no feedback_params attribute for now.

            algorithm.execute("Initialize", config["detector_config"])

            test_params["image_features"] = {}

            # Assume save_features == False and skip for now.

            round_id = 0
            end_of_dataset = False

            while not end_of_dataset:
                logging.info(f"Beginning round {round_id}")
                file_list = self.interface.dataset_request(
                    session_id, test_id, round_id
                )

                if file_list is not None:
                    # TODO: Saved features (assume not using for now).
                    features_dict, logits_dict = algorithm.execute(
                        "FeatureExtraction", file_list
                    )

                    results = {}
                    results["detection"] = algorithm.execute(
                        "WorldDetection", features_dict, logits_dict, red_light
                    )
                    results["classification"] = algorithm.execute(
                        "NoveltyClassification", features_dict, logits_dict
                    )

                    if config["use_feedback"]:
                        algorithm.execute("NoveltyAdaption", None)

                    results["characterization"] = algorithm.execute(
                        "NoveltyCharacterization", features_dict
                    )
                else:
                    end_of_dataset = True

                round_id += 1

        self.interface.terminate_session(session_id)
