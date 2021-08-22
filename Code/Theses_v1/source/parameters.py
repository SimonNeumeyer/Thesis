from pathlib import Path
from train import Trainer
from util import uuid
from settings import Settings
import json
import os

def load_settings(filepath):
    with open(filepath, "r") as file:
        settings = json.loads(file.read())
    return settings

if __name__ == "__main__":
    path = Path("../parameters")
    for file in os.listdir(path):
        settings = Settings(**load_settings(path / file))
        id = uuid()
        visualization_path = Path(f"../runs/{id}")
        settings.update(**{"org": {"visualization": {"fileName": id}}})
        os.rename(path / file, path / (id + ".json"))
        trainer = Trainer(**settings.settings)
        trainer.run()