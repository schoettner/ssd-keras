import json


class Params(object):

    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
