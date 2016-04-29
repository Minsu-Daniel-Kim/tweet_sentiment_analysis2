
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_data(filename, return_labels=False):

    logging.debug("loading data from %s", filename)
    data = np.genfromtxt(open("../data/%s" % filename), delimiter=",", usecols=range(0, 1000), skip_header=True)

    if return_labels:
        labels = np.genfromtxt(open("../data/%s" % filename), delimiter=",", usecols=[1000], skip_header=True)

        return labels, data
    else:
        labels = np.zeros(data.shape[0])
        return data
