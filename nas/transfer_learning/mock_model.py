import json

class MockLayer:
    def __init__(self, config):
        self.config = config
        self.name = config['name']
    @classmethod
    def from_config(cls, config):
        return cls(config)
    def get_config(self):
        return self.config
    def get_weights(self):
        pass
    @property
    def weights(self):
        return []

class MockModel:
    def __init__(self, config):
        self.config = config['config']
        self.class_name = config['class_name']
        self.inputs = config['config']['input_layers']
        self.layers = [
            MockLayer.from_config(layer) for layer in config['config']['layers']
        ]
    @classmethod
    def from_json(cls, json_desc):
        config = json.loads(json_desc)
        return cls(config)
    def __repr__(self):
        return f"MockModel with {len(self.layers)} layers"
    def save_weights(self, path):
        #just a mock, don't do anything
        pass
    def get_config(self):
        return self.config
    def summary(self):
        #just a mock, don't do anything
        pass
    def load_weights(path, by_name=True, skip_mismatch=True):
        #just a mock, don't do anything
        pass
    def save(path):
        #just a mock, don't do anything
        pass
