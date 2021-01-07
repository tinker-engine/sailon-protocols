import os
from typing import List, Optional, Dict, Any
import uuid


class SessionLog():
    def __init__(self, *args, **kwargs):
        # TODO
        self.activities = {
            "time": None,
            "tests": {},
        }


class LocalInterface():
    def __init__(self, test_spec_dir: str):
        """
        Args:
            test_spec_dir: Path to test specification root directory containing
                subdirectories for the different protocols, domains, and
                novelty detector algorithms. Bottom-level directories contain
                csv files corresponding to test ids, with each CSV file
                containing image filepaths for the test.
        """
        self.test_spec_dir = test_spec_dir

    def new_session(
        self,
        test_ids: List[str],
        protocol: str,
        domain: str,
        novelty_detector_spec: str,
        hints: List[str]
    ):
        """
        In the existing code, the test ids are read from the config. The test
        ids are passed to `Interface.session_request`, which passes them to
        `FileProvider.new_session`, which checks that each test id has an
        associated csv file. The csv file contains a list of newline-separated
        paths to image files.

        Args:
            test_ids: List of test ids with corresponding CSV files.
            protocol: Protocol name (eg, "OND").
            domain: Domain from config file (eg, "image_classification").
            novelty_detector_spec: Novelty detector version + novelty detector
                class (eg, "1.0.0.OND_5_14_A1" if the config contains
                `"novelty_detector_class": OND_5_14_A1"`).
            hints: Test-related hint/parameter (["red_light"]).
        """
        csv_dir = os.path.join(self.test_spec_dir, protocol, domain)

        for test_id in test_ids:
            fpath = os.path.join(csv_dir, f"{test_id}_single_df.csv")
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"No matching file found for test id {test_id}"
                )

        session_id = str(uuid.uuid4())

        # Log session information with SessionLog object for now instead of
        # writing to and reading from a log file as in legacy sail-on-api.
        #log_session()
        self.session_log = SessionLog(
            session_id=session_id,
            activity="created",
            content= {
                "protocol": protocol,
                "domain": domain,
                "detector": novelty_detector_spec,
                "hints": hints,
            }
        )
