import sys
sys.path.append(".")

import itertools

from dummy_interface import DummyInterface
from mock import MockDetector

import tinker


class ONDProtocol(tinker.protocol.Protocol):
    def __init__(self):
        super().__init__()
        self.interface = DummyInterface()

        # The `toolset` approach will ultimately be replaced.
        self.toolset = {}

    def get_config(self):
        return {
            "domain": "image_classification",
            "test_ids": [],
            "novelty_detector_class": "OND_5_14_A1",
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

        # Use toolset for now for initial testing.
        self.toolset.update(config)

        # The ultimate goal is to let smqtk handle the algorithm.
        algorithm = MockDetector(self.toolset)

        session_id = self.interface.new_session(
            test_ids=config["test_ids"], protocol="OND",
            domain=config["domain"],
            novelty_detector_spec="1.0.0.Mock",
            hints=config["hints"]
        )

        self.toolset["session_id"] = session_id

        for test_id in config["test_ids"]:
            self.toolset["test_id"] = test_id
            self.toolset["test_type"] = ""

            # Assume save_attributes == False and skip for now.
            # Assume no red light image for now.
            self.toolset["redlight_image"] = ""

            # Assume no feedback_params attribute for now.

            algorithm.execute(self.toolset, "Initialize")
            self.toolset["image_features"] = {}
            self.toolset["dataset_root"] = config["dataset_root"]
            self.toolset["dataset_ids"] = []

            # Assume save_features == False and skip for now.

            round_id = 0
            end_of_dataset = False

            while not end_of_dataset:
                self.toolset["round_id"] = round_id
                file_list = self.interface.dataset_request(
                    session_id, test_id, round_id
                )

                if file_list is not None:
                    # TODO: In legacy code, why is the file handle stored in
                    # toolset?
                    self.toolset["dataset"] = file_list
                    self.toolset["dataset_ids"].extend(file_list)

                    # TODO: Saved features.
                    x, y = algorithm.execute(self.toolset, "FeatureExtraction")
                else:
                    end_of_dataset = True

                round_id += 1
