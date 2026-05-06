import yaml
import os
import logging
import sys
import shutil
from typing import Tuple
from datetime import datetime

def parse(file: str) -> dict:
    """
    Parse a YAML file and return the contents as a dictionary.

    Args:
        file (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Raises:
        yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    with open(file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            return params
        except yaml.YAMLError as exc:
            print(exc)

def find_run(dir_snippet: str) -> Tuple[str, bool]:
    """
    Find a run directory that matches the given directory snippet.

    Args:
        dir_snippet (str): The snippet to match against the directory names.

    Returns:
        Tuple[str, bool]: A tuple containing the matching directory path and a boolean
        indicating whether a match was found.

    Raises:
        SystemExit: If multiple runs are found with the same name.

    """
    dir = ['output/' + x for x in os.listdir('output/')
           if dir_snippet in x and not os.path.islink('output/' + x)]
    if len(dir) > 1:
        logging.info("Multiple runs found with the same name")
        sys.exit()
    elif len(dir) == 0:
        return None, False
    else:
        logging.info("Warning: Continuing in an existing directory")
        return dir[0], True

def setup_dir(file: str) -> str:
    """
    Set up the directory structure for the training.

    Args:
        file (str): The path to the input file.

    Returns:
        str: The full path to the created directory.
    """
    params = parse(file)
    full_run_name, run_exists = find_run(params['name'])
    if not run_exists:
        now = datetime.now()
        full_run_name = "output/" + params['name'] + "_" + now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(full_run_name, exist_ok=True)
    os.makedirs(full_run_name+'/models/', exist_ok=True)
    latest_link = "output/" + params['name'] + "_latest"
    if os.path.islink(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.basename(full_run_name), latest_link)
    try:
        shutil.copy(file, full_run_name)
    except shutil.SameFileError:
        pass
    return full_run_name

def log_yaml(params: dict, source: str = None) -> None:
    """Log a (nested) YAML/config dict via logging.info, one key per line."""
    def _walk(d, indent=0):
        pad = "  " * indent
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    logging.info(f"{pad}{k}:")
                    _walk(v, indent + 1)
                elif isinstance(v, list):
                    logging.info(f"{pad}{k}: {v}")
                else:
                    logging.info(f"{pad}{k}: {v}")
        else:
            logging.info(f"{pad}{d}")

    header = f"=== YAML config: {source} ===" if source else "=== YAML config ==="
    logging.info(header)
    _walk(params)
    logging.info("=" * len(header))
