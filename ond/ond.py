import json
import pathlib

from config import default_config

# TODO: this import is a placeholder
#import tinker_engine


class ONDProtocol():
    def __init__(self, config, interface):
        """
        Args:
            config (dict|str): Dict representing pre-populated config or a
                path to a config file.

        .. todo:: if `config` is a file, it's assumed to be JSON for now.
            This will be updated alongside development of tinker engine
            config handling.
        """
        self.config = default_config()

        if isinstance(config, str):
            config_fpath = pathlib.Path(config).expanduser().resolve()
            if not config_fpath.exists():
                raise FileNotFoundError(f"File {config_fpath} does not exist")
            
            with open(config_fpath, "r") as f:
                config = json.load(f)

        if config is not None:
            self.config.update(config)

        # Get algorithm name from config.
        # TODO: rename this field in the config?
        novelty_algorithm_name = self.config["novelty_detector_class"]

        # TODO: This algorithm will ultimately be obtained via smqtk.
        # TODO: rename "detector_config" to "algorithm_config"?
        #NoveltyAlgorithm = tinker_engine.get_algorithm(novelty_algorithm_name)
        #self.algorithm = NoveltyAlgorithm(self.config["detector_config"])

        # TODO: get/instantiate interface from config? Or pass instance to
        # constructor?
        self.interface = interface

    def run_protocol(self):
        # TODO: should test IDs be in the config or passed in a different way?
        # TODO: interface calls should match ParInterface/LocalInterface usage.
        sess_id = self.interface.session_request(self.config["test_ids"])

        for test_id in self.config["test_ids"]:
            # NOTE: Use test_params on a per-test basis instead of toolset?
            test_params = self.interface.get_test_metadata(sess_id, test_id)

            # TODO: Handle save_attributes more robustly.
            if self.config["save_attributes"]:
                test_params["attributes"] = {}

            # TODO: Use a boolean "red_light" field?
            if "red_light" in test_params["metadata"]:
                pass
            else:
                pass

            # TODO: how to handle image classification feedback? Existing
            # sail_on_client uses an ImageClassificationFeedback class that
            # uses the interface to obtain image feedback.
            if (
                "feedback_params" in self.config["detector_config"]
                and self.config["domain"] == "image_classification"
            ):
                # TODO: refactor all of this?
                feedback_params = self.config["detector_config"]["feedback_params"]
                
                first_budget = feedback_params["first_budget"]
                income_per_batch = feedback_params["income_per_batch"]
                max_budget = feedback_params["max_budget"]

                # TODO
                test_params["image_classification_feedback"] = None

            # TODO: remainder of test functionality.

if __name__ == "__main__":
    protcol = ONDProtocol(None, None)
