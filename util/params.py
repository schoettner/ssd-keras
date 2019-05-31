import json
import yaml


# https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
class Params(object):

    def __init__(self, file_path: str, is_yml: bool = False):
        if is_yml:
            self.__update_yml__(file_path)
        self.__update_json__(file_path)

    def __update_json__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __update_yml__(self, yml_path):
        with open(yml_path) as f:
            params = yaml.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
