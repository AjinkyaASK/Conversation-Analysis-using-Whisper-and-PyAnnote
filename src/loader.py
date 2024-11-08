import threading
import itertools
import sys
import time

class Loader:
    def __init__(self, message="Loading...", delay=0.1):
        self.message = message
        self.delay = delay
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate, daemon=True)

    def _animate(self):
        spinner = itertools.cycle(["|", "/", "-", "\\"])
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{self.message} {next(spinner)}")
            sys.stdout.flush()
            time.sleep(self.delay)
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

    def start(self):
        self.stop_event.clear()
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
