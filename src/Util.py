import ast
import os
import datetime
import shutil

from src.lib.AssertSpec import AssertSpec
import numpy as np


def create_new_dir(basedir: str, prefix='', suffix='') -> str:
    dirname = str(int((datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds()))
    newdir = os.path.join(basedir, prefix + dirname + suffix)
    os.makedirs(newdir)
    return newdir


def getdims(array):
    dims = 0
    if isinstance(array, (np.ndarray, list)):
        dims += 1
        dims = dims + getdims(array[0])
    return dims


def compute_diff(x, y):
    if isinstance(x, (list, np.ndarray)) and len(x) == 0 and isinstance(y, (list, np.ndarray)) and len(y) == 0:
        return 0
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return np.abs(y)
    if isinstance(y, (list, np.ndarray)) and len(y) == 0:
        return np.abs(x)

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        diffs = np.abs([compute_diff(xx, yy) for xx, yy in zip(x, y)])
    elif isinstance(x, (list, np.ndarray)):
        diffs = np.abs([compute_diff(xx, y) for xx in x])
    elif isinstance(y, (list, np.ndarray)):
        diffs = np.abs([compute_diff(x, yy) for yy in y])
    else:
        diffs = [abs(x - y)]

    return np.abs(diffs)


def compute_max_diff(x, y):
    if isinstance(x, (list, np.ndarray)) and len(x) == 0 and isinstance(y, (list, np.ndarray)) and len(y) == 0:
        return 0
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return np.max(np.abs(y))
    if isinstance(y, (list, np.ndarray)) and len(y) == 0:
        return np.max(np.abs(x))

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):

        diffs = np.max(np.abs([compute_max_diff(xx, yy) for xx, yy in zip(x, y)]))
    elif isinstance(x, (list, np.ndarray)):
        diffs = np.max(np.abs([compute_max_diff(xx, y) for xx in x]))
    elif isinstance(y, (list, np.ndarray)):
        diffs = np.max(np.abs([compute_max_diff(x, yy) for yy in y]))
    else:
        diffs = np.max([abs(x - y)])

    return np.max(np.abs(diffs))


# assuming samples are tuples
def samples_stat(samples, assertSpec: AssertSpec):
    actual = [s[0] for s in samples]
    expected = [s[1] for s in samples]
    try:
        output = "Assert: {4}\nStats::\nActual:: min: {0}, max: {1}\nExpected:: min: {2}, max: {3}\n".format(
            np.min(actual), np.max(actual), np.min(expected), np.max(expected), assertSpec.print_spec())
    except ValueError:
        return "Empty array"
    return output


def sample_to_str(s):
    if isinstance(s, list) or isinstance(s, np.ndarray):
        return "[{0}]".format(','.join([sample_to_str(i) for i in s]))
    elif isinstance(s, np.ndarray):
        return np.array2string(s, max_line_width=np.inf,
                               separator=',',
                               threshold=np.inf).replace('\n', '')
    else:
        return str(s)


def get_name(node):
    if isinstance(node, ast.Attribute):
        return get_name(node.attr)
    if isinstance(node, ast.Subscript):
        return get_name(node.value)
    if isinstance(node, ast.Call):
        return ""
    if isinstance(node, ast.Name):
        return node.id
    assert isinstance(node, str), "type: %s" % type(node)
    return node  # should be str


def save_temp(source, target):
    shutil.copy(source, target)


def restore(source, target):
    shutil.move(source, target)
