import csv
import datetime
import io
from typing import List, Dict, IO
import uuid


class Session():
    def __init__(self, session_id, protocol, domain, detector, hints=None):
        self.session_id = session_id
        self.created = {
            "time": str(datetime.datetime.now()),
            "protocol": protocol,
            "domain": domain,
            "detector": detector,
            "hints": hints if hints is not None else [],
        }
        self.termination = False


class DummyInterface():
    """
    The DummyInterface class is modeled after the LocalInterface class from
    the existing legacy sail-on-client.
    """
    def __init__(self):
        self.session = None

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
        paths to image files. This class mocks the csv file's contents and
        stores session information in a Session object instead of a log file.

        Args:
            test_ids: List of test ids with corresponding CSV files.
            protocol: Protocol name (eg, "OND").
            domain: Domain from config file (eg, "image_classification").
            novelty_detector_spec: Novelty detector version + novelty detector
                class (eg, "1.0.0.OND_5_14_A1" if the config contains
                `"novelty_detector_class": OND_5_14_A1"`).
            hints: Test-related hint/parameter (["red_light"]).
        """

        session_id = str(uuid.uuid4())

        # Log session information with Session object for now instead of
        # writing to and reading from a log file as in legacy sail-on-api.
        self.session = Session(
            session_id=session_id, protocol=protocol, domain=domain,
            detector=novelty_detector_spec, hints=hints
        )

        return session_id

    def get_test_metadata(
        self, session_id: str, test_id: str, api_call: bool = True
    ):
        """
        Legacy sail-on-api defines a list of approved metadata keys to return
        to the client if `api_call=True` (if `api_call=False`, all keys are
        returned). This former behavior is ignored here and all keys are
        always returned for the time being.
        """
        if self.session is None:
            raise RuntimeError("Invalid session; call Interface.new_session()")

        metadata = {
            "known_classes": 413,
            "max_novel_classes": 413,
            "protocol": "OND",
            "red_light": "example_images/image3.jpg",
            "round_size": 2,
        }
        return metadata

    def dataset_request(self, session_id: str, test_id: str, round_id: int):
        """
        This method mocks reading the csv file of image filenames (based on
        the session id and test id).

        Returns: file_list
            List of filepaths (all filepaths for the given `test_id` if
            `round_id` is None, or filepaths corresponding to the given integer
            `round_id` for the round size defined in the test metadata), or
            `None` if there are no files in the specified `round_id`.
        """
        metadata = self.get_test_metadata(session_id, test_id, api_call=False)

        # TODO: read/write file instead of hardcoding filenames.

        lines = [
            "example_images/image1.jpg",
            "example_images/image2.jpg",
            "example_images/image3.jpg",
            "example_images/image4.jpg",
        ]

        file_list = None

        start_idx = 0
        end_idx = len(lines)
        if round_id is not None:
            start_idx = round_id * metadata["round_size"]
            end_idx = start_idx + metadata["round_size"]

        if start_idx < len(lines):
            file_list = lines[start_idx:end_idx]

        # Refactor and update this functionality to match log_session for
        # data_request, test, and round activities from sail-on-api.
        # Currently, this assumes round_id is not None.
        self.session.data_request = {
            "time": str(datetime.datetime.now()),
            "tests": {
                test_id: {}
            },
        }

        return file_list

    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str]
    ):
        """
        Update session log with results from a round.
        """
        pass

    def terminate_session(self, session_id: str):
        self.session.termination = True
