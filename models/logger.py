import sys


class Logger(object):
    def __init__(self, filename):
        self._terminal = sys.stdout
        self._log = open(filename, "w")

    def write(self, message):
        self._terminal.write(message)
        self._terminal.flush()
        self._log.write(message)
        self._log.flush()

    def flush(self):
        pass

    @property
    def terminal(self):
        return self._terminal

    @property
    def log(self):
        return self._log


def start_logger(filename):
    sys.stdout = Logger(filename)


def stop_logger():
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
