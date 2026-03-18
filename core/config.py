import yaml

class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys):
        value = self.config
        for k in keys:
            value = value[k]
        return value