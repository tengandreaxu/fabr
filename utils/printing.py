import logging

logging.basicConfig(level=logging.INFO)


def print_header(message: str):
    logging.info("#" * 47)
    logging.info(f"\t {message}")
    logging.info("#" * 47)
