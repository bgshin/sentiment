from os.path import join, dirname, abspath
import csv
# import random
from collections import namedtuple


Datapoint = namedtuple("Datapoint", "phraseid sentenceid phrase sentiment")

def _iter_data_file(filename):
    filename = 'train.tsv'
    DATA_PATH = abspath(join(dirname(__file__), ".", "dataset"))
    path = join(DATA_PATH, filename)

    it = csv.reader(open(path, "r"), delimiter="\t")

    row = next(it)  # Drop column names

    if " ".join(row[:3]) != "PhraseId SentenceId Phrase":
        raise ValueError("Input file has wrong column names: {}".format(path))

    for row in it:
        if len(row) == 3:
            row += (None,)
        yield Datapoint(*row)


def iter_corpus(__cached=[]):
    """
    Returns an iterable of `Datapoint`s with the contents of train.tsv.
    """
    if not __cached:
        __cached.extend(_iter_data_file("train.tsv"))
    return __cached



data = list(iter_corpus())

print 'a'