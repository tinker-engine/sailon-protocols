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

        algo_config_params = config["detector_config"]

        # smqtk will ultimately handle retrieval of the algorithm.
        algorithm = RandomNoveltyDetector()

        session_id = self.interface.new_session(
            test_ids=config["test_ids"], protocol="OND",
            domain=config["domain"],
            novelty_detector_spec="1.0.0.Random",
            hints=config["hints"]
        )

        for test_id in config["test_ids"]:
            algo_test_params = {
                "dataset_ids": [],
                "dataset_root": config["dataset_root"],
                "image_features": {},
            }

            test_metadata = self.interface.get_test_metadata(
                session_id=session_id,
                test_id=test_id
            )

            algo_test_data = {}
            if "red_light" in test_metadata:
                algo_test_data["red_light_image"] = test_metadata.get(
                    "red_light", ""
                )

            algorithm.initialize(algo_config_params)

            round_id = 0
            end_of_dataset = False
            while not end_of_dataset:
                logger.info(f"Beginning round {round_id}")

                algo_test_data["round_id"] = round_id

                try:
                    dataset = self.interface.dataset_request(
                        session_id, test_id, round_id
                    )
                except RoundError:
                    end_of_dataset = True
                    continue

                with open(dataset, "r") as dataset_:
                    dataset_ids = dataset_.readlines()
                    image_ids = [image_id.strip() for image_id in dataset_ids]
                    algo_test_params["dataset_ids"].extend(image_ids)

                if config["use_saved_features"]:
                    # TODO
                    pass
                else:
                    (
                        algo_test_data["features_dict"],
                        algo_test_data["logits_dict"]
                    ) = algorithm.feature_extraction(algo_test_params)

                    # TODO: save features

                results = {}

                results["detection"] = algorithm.world_detection(
                    algo_test_params, algo_test_data
                )

                results["classification"] = algorithm.novelty_classification(
                    algo_test_params, algo_test_data
                )

                # TODO: post results

                if config["use_feedback"]:
                    algorithm.novelty_adaption(algo_test_params, algo_test_data)

                # TODO: round cleanup

                round_id += 1

            if config["save_features"]:
                # TODO: save features
                logging.info(
                    f"Writing features to {config['save_features']}"
                )

            # TODO: save attributes

            results["characterization"] = algorithm.novelty_characterization(
                algo_test_params, algo_test_data
            )

            # TODO: test cleanup

        self.interface.terminate_session(session_id)
