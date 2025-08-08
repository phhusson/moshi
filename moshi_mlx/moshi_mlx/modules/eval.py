import mlx.core as mx
import functools
from inspect import signature

def eval_mx_arrays(func):
    """
    Decorator that:
    - Calls mx.eval() on all input arguments of type mx.nd.NDArray before the function runs.
    - Calls mx.eval() on all return values of type mx.nd.NDArray (or tuple/list of NDArrays).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to parameter names
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Evaluate all input arguments that are mx.nd.NDArray
        for name, value in bound_args.arguments.items():
            if isinstance(value, mx.array):
                #mx.eval(value)
                mx.async_eval(value)
                pass

        # Call the original function
        result = func(*bound_args.args, **bound_args.kwargs)

        # Handle return values: evaluate if they are mx.nd.NDArray, list, or tuple of arrays
        if isinstance(result, mx.array):
            #mx.eval(result)
            mx.async_eval(result)
            pass
        elif isinstance(result, (list, tuple)):
            # Evaluate each NDArray in the list/tuple
            for item in result:
                if isinstance(item, mx.array):
                    #mx.eval(item)
                    mx.async_eval(item)
                    pass

        return result

    return wrapper
