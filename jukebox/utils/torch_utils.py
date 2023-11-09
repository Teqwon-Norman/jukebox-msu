import gc
import torch as t

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None

def empty_cache():
    """
        Clear GPU memory by collecting garbage and emptying the CUDA memory cache.

        This function utilizes the `gc.collect()` method to perform garbage collection, 
        freeing up any unreferenced objects

        in the Python memory space. Additionally, it calls `torch.cuda.empty_cache()` 
        to release cached memory held by the CUDA memory allocator in PyTorch, providing 
        a way to explicitly clear GPU memory.

        Note: Use this function when you want to manually release GPU memory in a 
        PyTorch-based application, especially in situations where automatic garbage 
        collection may not be sufficient.
    """

    gc.collect()
    t.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())

