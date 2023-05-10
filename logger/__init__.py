import logging


class Logger:
    logger: logging.Logger
    fh: logging.FileHandler

    def __init__(self, name: str):
        self.logger = logging.getLogger(name.title())
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(name + ".log")
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(logging.Formatter('[%(name)s][%(asctime)s][%(levelname)s]: %(message)s'))
        self.logger.addHandler(self.fh)

    def error(self, message: str):
        self.logger.error(message)

    def log(self, message: str):
        self.logger.info(message)

    def warn(self, message: str):
        self.logger.warning(message)
