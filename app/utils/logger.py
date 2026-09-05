import logging
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
logs_dir = os.path.join(project_root, "logs")

os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "medical_generative.log")


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(process)d | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File only - everything (including exceptions/tracebacks logged via
    # logger.exception()) goes to logs/medical_generative.log so the
    # terminal running `streamlit run app.py` stays clean.
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger