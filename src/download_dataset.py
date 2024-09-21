import argparse
import logging
import logging.config

from src.definitions import LOGGING_CONFIG_PATH


def __empty_action(logger: logging.Logger) -> None:
    logger.info("Nothing to download")


MODEL_TO_ACTION_MAP = {"genetic_ipc_prediction": __empty_action}


def download_dataset(model_name: str, logger: logging.Logger) -> None:
    action = MODEL_TO_ACTION_MAP.get(model_name)

    if not action:
        raise RuntimeError("Model with name [ %s ] is not found" % model_name)
    else:
        action(logger)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument("model")

    args = parser.parse_args()
    model = args.model

    download_dataset(model, logging.getLogger(__name__))
