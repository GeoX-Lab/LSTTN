import time
import pickle
import time
import os


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print("%s: %0.8fs..." % (name, elapsed))
        return result

    return clocked


def load_pkl(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def dump_pkl(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


class TrainLogger:
    def __init__(self):
        path = "log/"
        cur_time = time.strftime("%Y%m%d %H%M%S", time.localtime())
        cur_time = cur_time.replace(" ", "-")
        try:
            os.makedirs(path + cur_time)
        except FileExistsError:
            pass
