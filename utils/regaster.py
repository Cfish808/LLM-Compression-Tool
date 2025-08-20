class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self._storage = {}

    def __call__(self, obj):
        return self.add(obj)

    def add(self, obj):
        def insert(name, func):
            if not callable(func):
                raise TypeError(f"Invalid value: {func}, must be callable!")
            if name in self._storage:
                raise KeyError(f"{name} already registered.")
            self[name] = func
            return func

        if callable(obj):
            return insert(obj.__name__, obj)
        else:
            return lambda func: insert(obj, func)

    def __setitem__(self, name, func):
        self._storage[name] = func

    def __getitem__(self, name):
        return self._storage[name]

    def __contains__(self, name):
        return name in self._storage

    def __repr__(self):
        return repr(self._storage)

    def all_keys(self):
        return self._storage.keys()

    def all_values(self):
        return self._storage.values()

    def all_items(self):
        return self._storage.items()

MODEL_REGISTRY = Registry()
QUANTIZATION_REGISTRY = Registry()