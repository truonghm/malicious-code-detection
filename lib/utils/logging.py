import logging.config


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "WARNING": "33",  # yellow
        "INFO": "32",  # green
        "DEBUG": "36",  # cyan
        "CRITICAL": "35",  # magenta
        "ERROR": "31",  # red
    }

    def format(self, record: logging.LogRecord) -> str:
        colored_record = record
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, "37")  # default to white
        colored_levelname = f"\033[1;{seq}m{levelname}\033[1;0m"  # 1; - bright, 0m - reset
        colored_record.levelname = colored_levelname
        return super().format(colored_record)


def get_logger(logger_name: str, log_level: str = "INFO", to_file: bool = True) -> logging.Logger:
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": ColoredFormatter,
                "fmt": "%(asctime)s, %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "default": {
                "fmt": "%(asctime)s, %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "colored",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
    }

    if to_file:
        log_config["handlers"]["file"] = {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": f"{logger_name}.log",
            "mode": "a",
        }
        log_config["loggers"] = {
            logger_name: {"handlers": ["default", "file"], "level": log_level},
        }

    else:
        log_config["loggers"] = {
            logger_name: {"handlers": ["default"], "level": log_level},
        }

    logging.config.dictConfig(log_config)
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    return logger


logger = get_logger("js-code-detection", log_level="DEBUG", to_file=True)
