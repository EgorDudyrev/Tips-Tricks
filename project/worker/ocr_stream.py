import logging
from threading import Thread

from cnd.ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader
from cnd.ocr.converter import strLabelConverter
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from datetime import datetime

CV_CONFIG = Config(CONFIG_PATH)


class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader, model_path):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.video_reader = video_reader
        self.ocr_thread = None
        self.time_start = datetime.now()
        self.stopped = False

        converter = strLabelConverter(CV_CONFIG.get('alphabet'))
        self.predictor = Predictor(model_path, converter) #TODO: Your Predictor


        self.logger.info("Create OcrStream")

    def _ocr_loop(self):
        try:
            while True:
                if self.stopped:
                    return
                frame = self.video_reader.read()

                pred = self.predictor.predict(frame[None])
                self.state.text = pred
                self.state.frame = frame
                self.state.frame_id += 1
                if self.state.frame_id % 24 == 0:
                    cur_t = datetime.now()
                    secs = (cur_t-self.time_start).seconds
                    self.logger.info(f"FPS: {self.state.frame_id/secs if secs>0 else 0}")

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
        self.stopped = True
        if self.ocr_thread is not None:
            self.ocr_thread.join()
        self.logger.info("Stop OcrStream")
