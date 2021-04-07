import itertools
import logging

from sailon import DummyInterface, RandomNoveltyDetector
import tinker


logger = logging.getLogger(__name__)


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
        algorithm = RandomNoveltyDetector()

        session_id = self.interface.new_session(
            test_ids=config["test_ids"], protocol="OND",
            domain=config["domain"],
            novelty_detector_spec="1.0.0.Random",
            hints=config["hints"]
        )

        for test_id in config["test_ids"]:
            test_metadata = self.interface.get_test_metadata(
                session_id=session_id,
                test_id=test_id
            )

            test_params = {}

            red_light = test_metadata.get("red_light", "")

            # Assume no feedback_params attribute for now.

            algorithm.execute("Initialize", config["detector_config"])

            test_params["image_features"] = {}

            round_id = 0
            end_of_dataset = False

            while not end_of_dataset:
                logger.info(f"Beginning round {round_id}")
                file_list = self.interface.dataset_request(
                    session_id, test_id, round_id
                )

                if file_list is not None:
                    features_dict, logits_dict = algorithm.execute(
                        "FeatureExtraction", file_list
                    )

                    results = {}
                    results["detection"] = algorithm.execute(
                        "WorldDetection", features_dict, logits_dict,
                        red_light, round_id=round_id
                    )
                    results["classification"] = algorithm.execute(
                        "NoveltyClassification", features_dict, logits_dict,
                        round_id=round_id
                    )

                    if config["use_feedback"]:
                        algorithm.execute("NoveltyAdaption", None)

                    if config["save_features"]:
                        logger.info(
                            f"Writing features to {config['save_features']}"
                        )

                    results["characterization"] = algorithm.execute(
                        "NoveltyCharacterization", features_dict, logits_dict
                    )
                else:
                    end_of_dataset = True

                round_id += 1

        self.interface.terminate_session(session_id)
