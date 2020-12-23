import json
import pathlib


def default_config():
    config = {
        "detector_config": {
            "feature_extractor_params": {
                "backbone_weight_path": "",
                "name": "i3d",
                "arch": "i3d-50",
                # TODO
            },
        },
    }
    return config


class ONDProtocol():
    def __init__(self, config):
        """
        Args:
            config (dict|str): Dict representing pre-populated config or a
                path to a config file.

        .. todo:: if `config` is a file, it's assumed to be JSON for now.
            This will be updated alongside development of tinker engine
            config handling.
        """
        if isinstance(config, str):
            config_fpath = pathlib.Path(config).expanduser().resolve()
            if not config_fpath.exists():
                raise FileNotFoundError(f"File {config_fpath} does not exist")
            
            with open(config_fpath, "r") as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("config must be dict or path to config file")

        # TODO: get default config from `default_config` and update it with
        # the options present in `config`?

        # Get algorithm name from config.
        # TODO: rename this field in the config?
        novelty_algorithm_name = self.config["novelty_detector_class"]

        # TODO: import/load the algorithm class via tinker/smqtk?
        # TODO: rename "detector_config" to "algorithm_config"?
        NoveltyAlgorithm = foo.get_algorithm(novelty_algorithm_name)
        self.algorithm = NoveltyAlgorithm(self.config["detector_config"])

    def run_protocol(self):
        pass

    def run_test(self, test_params):
        """
        TODO: call this from run_protocol?
        """
        pass
