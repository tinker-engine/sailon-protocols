import csv
import datetime
import os
import tempfile
from typing import List, Dict
import uuid

import cv2
import numpy as np

from .errors import RoundError


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

        # Create a temp directory and write several garbage images to it.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_fpaths = []

        image_w, image_h = (512, 512)
        num_images = 12
        for i in range(num_images):
            img = np.random.randint(
                0, 255, (image_h, image_w, 3), dtype=np.uint8
            )
            fpath = os.path.join(self.temp_dir.name, f"image{i}.png")
            cv2.imwrite(fpath, img)
            self.image_fpaths.append(fpath)


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
    ) -> str:
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

        Returns:
            uuid corresponding to the new session
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
            "round_size": 16,
        }
        return metadata

    def dataset_request(
        self, session_id: str, test_id: str, round_id: int
    ) -> str:
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

        # Get list of images created as part of this session.
        image_fpaths = self.session.image_fpaths

        file_list = None

        start_idx = 0
        end_idx = len(image_fpaths)
        if round_id is not None:
            start_idx = round_id * metadata["round_size"]
            end_idx = start_idx + metadata["round_size"]

        if start_idx < len(image_fpaths):
            file_list = image_fpaths[start_idx:end_idx]

        # To match existing functionality, write the list of files to a text
        # file and return the path of the text file.
        if file_list is None:
            # End of dataset; no more files/rounds.
            raise RoundError
        else:
            temp_dir = self.session.temp_dir.name
            file_list_fname = f"round_{round_id}_file_list.csv"
            file_list_fpath = os.path.join(temp_dir, file_list_fname)

            file_list = ["/home/najam/sailon/videos/v_PizzaTossing_g03_c04.avi"]

            with open(file_list_fpath, "w") as f:
                for image_fpath in file_list:
                    f.write(image_fpath + "\n")

        # Refactor and update this functionality to match log_session for
        # data_request, test, and round activities from sail-on-api.
        # Currently, this assumes round_id is not None.
        self.session.data_request = {
            "time": str(datetime.datetime.now()),
            "tests": {
                test_id: {}
            },
        }

        return file_list_fpath

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
