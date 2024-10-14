import json
import os
from xml.etree import ElementTree

import pandas as pd
from loguru import logger


def read_csv(filepath, filename):
    return pd.read_csv(os.path.join(filepath, filename))


def read_json(filepath, filename):
    return json.load(open(os.path.join(filepath, filename), "rb"))


def save_csv(content, filepath, filename, index=False):
    content.to_csv(os.path.join(filepath, filename), index=index)


def read_parquet(filepath, filename):
    return pd.read_parquet(os.path.join(filepath, filename))


def read_xml_file(file_path):
    try:
        tree = ElementTree.parse(file_path)

        root = tree.getroot()

    except ElementTree.ParseError as e:
        logger.info(f"Error parsing the XML file: {e}")
    except Exception as e:
        logger.info(f"An error occurred: {e}")

    return ElementTree.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
