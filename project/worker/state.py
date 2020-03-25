import threading
from datetime import datetime
from copy import copy


class State:
    def __init__(self):
        self.exit_event = threading.Event()
        self.text = ""
        self.frame_id = 0
        self.time_start = datetime.utcnow().timestamp()
        self.frame = None

    @property
    def data(self):
        #data = copy(self.text)
        data = {'text': copy(self.text), 'ts': (datetime.utcnow().timestamp()-self.time_start),
                'frame_id': self.frame_id}
        #data['ts'] = datetime.utcnow().timestamp()
        return data
