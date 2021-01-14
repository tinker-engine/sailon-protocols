import tinker


class ONDProtocol(tinker.protocol.Protocol):
    def __init__(self):
        pass

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
