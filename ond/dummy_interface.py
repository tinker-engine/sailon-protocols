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

        # This is the metadata from OND.47392.000000.8901_metadata.json.
        # For initial testing, red light will be ignored.
        # TODO: Reduce round size for testing?
        metadata = {
            "known_classes": "413",
            "max_novel_classes": "413",
            "protocol": "OND",
            #"red_light": "test/test_images/images_205000_to_210000/205023.jpeg",
            "round_size": 128
        }
        return metadata

    def dataset_request(session_id: str, test_id: str, round_id: int) -> IO:
        """
        This method mocks reading the csv file of image filenames (based on
        the session id and test id).
        """
        metadata = self.get_test_metadata(session_id, test_id, api_call=False)

        # Ignore `round_id` for now and simply return a file handle for the
        # whole file instead of a chunk of the csv file.
        if round_id is not None:
            # Create a temporary file handle pointing to a temp file that
            # contains only the portion of the test id csv file containing
            # the chunk of filenames for the current round.
            #round_pos = int(round_id) * int(metadata["round_size"])
            pass

        fpath = "sample_test.csv"
        file_ = open(fpath, "rb")

        log_session(
            self.results_folder,
            session_id,
            test_id=test_id,
            round_id=round_id,
            activity="data_request"
        )

        # Refactor and update this functionality to match log_session for
        # data_request, test, and round activities from sail-on-api.
        # Currently, this assumes round_id is not None.
        self.session.data_request = {
            "time": str(datetime.datetime.now()),
            "tests": {
                test_id: {}
            },
        }

        return file_

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

    def terminate_tession(self, session_id: str):
        self.session.termination = True
