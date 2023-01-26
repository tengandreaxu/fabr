import json
import pickle


def save_pickle(obj, file_name: str):

    with open(file_name, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_dict_to_json(dict_: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dict_, f)
