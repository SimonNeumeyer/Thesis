from util import Constants, path_tensorboard, uuid

class Settings():

    def __init__(self, **kwargs):
        self.settings = {}
        self._init_default()
        self.update(**kwargs)

    def _init_default(self):
        self.settings = {
            "model": {
                "features": 0,
                "classes": 0,
                "darts": {
                    "apply": True,
                    "active": True,
                    "sampling": False,
                    "reduce": Constants.REDUCE_FUNC_SUM,
                    "normalize": False, #not implemented
                    "sharedWeights": False,
                    "randomInit": True
                },
                "graphs": {
                    "features": 17,
                    "reduce": Constants.REDUCE_FUNC_SUM,
                    "normalize": False, #not implemented
                },
                "cells": [
                    {
                        "numberNodes": 5,
                        "numberGraphs": 5
                    },
                    {
                        "numberNodes": 5,
                        "numberGraphs": 5
                    },
                    {
                        "numberNodes": 4,
                        "numberGraphs": 2
                    }
                ]
            },
            "optimization": {
                "epochs": 23,
                "optimizer": Constants.OPTIMIZER_ADAM,
                "batchSize": 61
            },
            "org": {
                "visualization": {
                    "visualize": True,
                    "fileName": uuid()
                }
            }
        }

    def __getitem__(self, key):
        assert key in self.settings, f"Setting '{key}' not available in {self.settings}"
        return self.settings[key]

    def update (self, _path=[], **kwargs):
        view = self.settings
        for i in _path:
            view = view[i]
        for key in kwargs:
            assert key in view, f"Setting '{key}' not available in {view}"
            assert isinstance(kwargs[key], dict) == isinstance(view[key], dict), f"Different types of {view} and {kwargs} for key '{key}'"
            if isinstance(kwargs[key], dict):
                rec_path = _path.copy()
                rec_path.append(key)
                self.update(_path=rec_path, **kwargs[key])
            else:
                view[key] = kwargs[key]