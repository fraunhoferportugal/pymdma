from os.path import join

from config import project_dir
from loguru import logger

with open(join(project_dir, "VERSION"), encoding="utf-8") as f:
    __version__ = f.read()


if __name__ == "__main__":
    logger.info(__version__)

# EOF
