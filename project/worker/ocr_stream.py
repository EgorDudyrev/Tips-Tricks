import logging
from threading import Thread

from cnd.ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader


class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.video_reader = video_reader
        self.ocr_thread = None

        self.predictor = #TODO: Your Predictor

        self.logger.info("Create OcrStream")

    def _ocr_loop(self):
        try:
            frame = self.video_reader.read()
            pred = self.predictor.predict([frame])
            self.state.text = pred

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def _start_ocr(self):
        self.ocr_thread = Thread(target=self._ocr_loop)
        self.ocr_thread.start()

    def start(self):
        self._start_ocr()
        self.logger.info("Start OcrStream")

    def stop(self):
        if self.ocr_thread is not None:
            self.ocr_thread.join()
        self.logger.info("Stop OcrStream")
