import logging
import time
from threading import Thread

import cv2
from worker.state import State
from worker.video_reader import VideoReader
from worker.video_writer import VideoWriter

from textdistance import levenshtein
import re


class Visualizer:
    def __init__(self, state: State, coord, color=(0, 0, 255), thick=2, font_scale=1.2, font=cv2.FONT_HERSHEY_SIMPLEX,
                 true_text=''):
        self.state = state
        self.coord_x, self.coord_y = coord
        self.color = color
        self.thickness = thick
        self.font_scale = font_scale
        self.font = font
        self.true_text = true_text
        self.acc_best = 0
        self.best_text = ''

    def _draw_ocr_text(self):
        text = self.state.text
        frame = self.state.frame

        text = text+'\n' if text is not None else ''

        if self.true_text is not '':
            acc = sum([a == b for a, b in zip(text, self.true_text)])/len(self.true_text)
            if acc > self.acc_best:
                self.acc_best = acc
                self.best_text = text
            text += f"best prediction: {self.best_text}\n"
            text += f"accuracy per letter: {round(self.acc_best,4)}\n"
            text += f"levenshtein: {levenshtein.distance(self.best_text, self.true_text)}\n"

        text += f"FPS: {round(self.state.frame_id/self.state.data['ts'],2)}"
        text = re.sub('\n+', '\n', text)

        for idx, txt in enumerate(text.split('\n')):
            #TODO: Put text on frame
            cv2.putText(frame, txt, (self.coord_x, self.coord_y+idx*50), self.font,
                        self.font_scale, self.color, self.thickness)


        return frame

    def __call__(self):
        frame = self._draw_ocr_text()
        return frame


class VisualizeStream:
    def __init__(self, name,
                 in_video: VideoReader,
                 state: State, video_path, fps, frame_size, coord, true_text=''):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.coord = coord
        self.fps = fps
        self.frame_size = tuple(frame_size)

        self.out_video = VideoWriter("VideoWriter", video_path, self.fps, self.frame_size)
        self.sleep_time_vis = 1. / self.fps
        self.in_video = in_video
        self.stopped = True
        self.visualize_thread = None

        self.visualizer = Visualizer(self.state, self.coord, true_text=true_text)

        self.logger.info("Create VisualizeStream")

    def _visualize(self):
        try:
            while True:
                if self.stopped:
                    return
                #TODO: Read && resize (if needed) then use visualizer to put text on frame
                # then save video with VideoWriter
                frame = self.visualizer()
                if frame is not None:
                    #continue
                    frame = cv2.resize(frame, self.frame_size)
                    self.out_video.write(frame)

                time.sleep(self.sleep_time_vis)

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def start(self):
        self.logger.info("Start VisualizeStream")
        self.stopped = False
        self.visualize_thread = Thread(target=self._visualize, args=())
        self.visualize_thread.start()

    def stop(self):
        self.logger.info("Stop VisualizeStream")
        self.stopped = True
        self.out_video.stop()
        if self.visualize_thread is not None:
            self.visualize_thread.join()
