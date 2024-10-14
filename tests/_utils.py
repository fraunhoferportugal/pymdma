def prune_params(params):
    return {key: value for key, value in params.items() if value is not None}
