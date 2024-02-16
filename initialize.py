import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from ruamel.yaml import YAML
import os

str2level = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def config_logger(
    name: str,
    level="INFO",
    to_console=True,
    save_path=None,
    mode="w",
    max_bytes=1048576,
    backup_count=3,
) -> None:
    logger = logging.getLogger(name=name)
    logger.setLevel(level=str2level.get(level, logging.WARNING))
    formatter = logging.Formatter(
        "[%(funcName)s][%(levelname)s] >>>>> %(message)s"
    )
    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level=str2level.get(level, logging.WARNING))
        logger.addHandler(console)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        rotating_file = ConcurrentRotatingFileHandler(
            filename=save_path,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        rotating_file.setFormatter(formatter)
        rotating_file.setLevel(level=str2level.get(level, logging.WARNING))
        logger.addHandler(rotating_file)


def init():
    config = YAML().load(open("./config/logger.yml", "r"))
    for k, v in config.items():
        config_logger(name=k, **v)
