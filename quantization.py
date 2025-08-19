

class QuantizationBase():
    def __init__(self, **kwargs):
        self.tokenizer = tokenizer
        self.model = model
        pass

    def get_loader(self):
        pass

    def quantize(self):
        pass

    def eval(self):
        pass

    def save(self):
        pass