"""Bounded diagnostic progress; never stores game state or changes scheduling."""
from contextlib import contextmanager
import json
import threading
import time


class Progress:
    def __init__(self, path, interval=10):
        self.path=path;self.interval=interval;self.started=time.monotonic()
        self.lock=threading.Lock();self.done=threading.Event();self.lanes={};self.error=None
        self.thread=threading.Thread(target=self._loop,name='b02-progress',daemon=True)

    def update(self, key, stage, **details):
        with self.lock:
            self.lanes[str(key)]=dict(stage=stage,updated_seconds=time.monotonic()-self.started,**details)

    @contextmanager
    def stage(self, key, stage, **details):
        self.update(key,stage,**details)
        try:yield
        except BaseException as error:
            with self.lock:
                # An outer stage must retain the more specific inner failure.
                if self.lanes[str(key)]['stage']!='failed':
                    self.lanes[str(key)]=dict(stage='failed',operation=stage,error=str(error),
                        updated_seconds=time.monotonic()-self.started,**details)
            raise

    def flush(self):
        with self.lock:
            value=dict(elapsed_seconds=time.monotonic()-self.started,lanes=dict(self.lanes))
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temp.replace(self.path)
        print('B02 progress: '+json.dumps(value,separators=(',',':')),flush=True)

    def _loop(self):
        try:
            while not self.done.wait(self.interval):self.flush()
        except BaseException as error:self.error=error

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self, kind, value, traceback):
        self.done.set();self.thread.join();self.flush()
        if kind is None and self.error is not None:raise self.error
