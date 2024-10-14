from typing import List, Union


def set_property(key: str, value: Union[List[str], str]):
    """Create a decorator that sets a specific property of the decorated
    function to a given value.

    Parameters
    ----------
    key : str
        The name of the property to be set.
    value : Any
        The value to be assigned to the specified property key.

    Returns
    -------
    decorator : function
        A decorator function that sets the specified property of a decorated function.
    """

    def decorator(func):
        setattr(func, key, value)
        return func

    return decorator


def inherit_annotations(cls):
    annotations = {}
    for parent in cls.__bases__:
        annotations.update(getattr(parent, "__annotations__", {}))
    cls.__annotations__.update(annotations)
    return cls
