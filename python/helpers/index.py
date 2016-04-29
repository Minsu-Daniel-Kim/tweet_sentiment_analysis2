__author__ = 'DanielMinsuKim'

from helpers.utils import load_data
import logging

logger = logging.getLogger(__name__)


def main():

    logger.info("loading data....")
    load_data("train100000.csv")







if __name__ == '__main__':
    main()