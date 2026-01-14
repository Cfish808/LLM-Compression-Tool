class DotDict(dict):
    """
    支持 obj.key 访问的 dict
    """
    __slots__ = ()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]



def to_dotdict(obj):
    """
    递归将 dict 转换为 DotDict
    """
    if isinstance(obj, dict):
        return DotDict({k: to_dotdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_dotdict(v) for v in obj]
    else:
        return obj


def flatten_dict(d):
    """
    将嵌套 dict 展平为单层 dict
    """
    items = {}
    for k, v in d.items():
        new_key =  k
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[new_key] = v
    return items
