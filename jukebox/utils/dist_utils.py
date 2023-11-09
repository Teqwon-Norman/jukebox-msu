import os
import torch
import jukebox.utils.dist_adapter as dist

from time import sleep
from mpi4py import MPI  # This must be imported in order to get errors from all ranks to show up

def print_once(msg):
    if (not dist.is_available()) or dist.get_rank()==0:
        print(msg)

def print_all(msg):
    """
        Print a message to the console, either on all processes or only on a specific subset.

        Parameters:
        - msg (str): The message to be printed.

        This function checks if the distributed training environment is available 
        using PyTorch's `dist` module.

        If not available, it prints the message on all processes. If available, it 
        prints the message only on processes.

        with ranks divisible evenly by 8. This is often useful to reduce the amount of 
        printed output in distributed setups.

        Note: Ensure that the function is called within a distributed training context for 
        accurate rank-based printing.
    """

    if not dist.is_available():
        print(msg)

    elif dist.get_rank() % 8 == 0:
        print(f'{dist.get_rank() // 8}: {msg}')

def allgather(x):
    xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    xs = torch.cat(xs, dim=0)
    return xs

def allreduce(x, op=dist.ReduceOp.SUM):
    x = torch.tensor(x).float().cuda()
    dist.all_reduce(x, op=op)
    return x.item()

def allgather_lists(xs):
    bs = len(xs)
    total_bs = dist.get_world_size()*len(xs)
    lengths = torch.tensor([len(x) for x in xs], dtype=t.long, device='cuda')
    lengths = allgather(lengths)
    assert lengths.shape == (total_bs,)
    max_length = torch.max(lengths).item()

    xs = torch.tensor([[*x, *[0]*(max_length - len(x))] for x in xs], device='cuda')
    assert xs.shape == (bs, max_length), f'Expected {(bs, max_length)}, got {xs.shape}'
    xs = allgather(xs)
    assert xs.shape == (total_bs,max_length), f'Expected {(total_bs, max_length)}, got {xs.shape}'

    return [xs[i][:lengths[i]].cpu().numpy().tolist() for i in range(total_bs)]

def setup_dist_from_mpi(
    master_addr: str = "127.0.0.1",
    backend: str = "nccl",
    port: int = 29500,
    n_attempts: int = 5,
    verbose: bool = False
) -> tuple[int, int, torch.device]:
    """
        Set up distributed training environment using MPI or fallback to single-node configuration.

        Parameters:
            - master_addr: 
                - The address of the master node for distributed training. Default is "127.0.0.1".
            - backend: The backend used for distributed training, e.g., "nccl". Default is "nccl".
            - port: The port to use for communication between nodes. Default is 29500.
            - n_attempts: Number of attempts to establish a distributed environment. Default is 5.
            - verbose: If True, print additional information during setup. Default is False.

        Returns:
            Tuple[int, int, torch.device]: 
                - A tuple containing the MPI rank, local rank, and the PyTorch device.
                - If distributed training is not available, it sets up a single-node
                environment with default values.

        High-level Explanation:
            This function sets up a distributed training environment using MPI, falling back 
            to a single-node configuration if MPI is not available. It initializes variables 
            such as master address, backend, port, and attempts to establish a 
            distributed environment. If unsuccessful, it configures a single-node environment with 
            GPU/CPU device information and returns the MPI rank, local rank, and PyTorch device.

        Note:
            Ensure that PyTorch and MPI are properly installed for distributed training to work.
    """

    if dist.is_available():
        return _setup_dist_from_mpi(master_addr, backend, port, n_attempts, verbose)

    use_cuda = torch.cuda.is_available()
    print(f'Using cuda {use_cuda}')

    mpi_rank = 0
    local_rank = 0

    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    torch.cuda.set_device(local_rank)

    return mpi_rank, local_rank, device

def _setup_dist_from_mpi(
    master_addr: str, backend: str, port: int, n_attempts: int, verbose: bool
) -> tuple[int, int, torch.device]:
    """
        Set up distributed training environment using MPI and PyTorch's `dist` module.

        Parameters:
            - master_addr: The address of the master node for initialization.
            - backend: 
                The backend to use for distributed training (e.g., 'nccl' for NVIDIA GPUs).
            - port: The port for communication during distributed training.
            - n_attempts: 
                The number of attempts to initialize NCCL, guarding against race conditions.
            - verbose: If True, print additional information during setup.

        Returns:
            - Tuple[int, int, torch.device]: 
                A tuple containing the MPI rank, local rank, and PyTorch device.

        - This function sets up the distributed training environment 
        using MPI and PyTorch's `dist` module.

        - It configures environment variables, such as RANK, WORLD_SIZE, MASTER_ADDR, 
        and MASTER_PORT, necessary for communication between distributed nodes.

        - The function attempts to initialize the process group with the specified backend
        and environment initialization method. In case of a runtime error 
        (common in large-scale distributed setups), it retries the initialization for a given number 
        of attempts while avoiding a potential race condition.

        - The function returns a tuple containing the MPI rank, local rank, and PyTorch device 
        for the current process.
    """

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    os.environ["RANK"] = str(mpi_rank)
    os.environ["WORLD_SIZE"] = str(mpi_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_SOCKET_NTHREADS"] = "8"

    # Pin this rank to a specific GPU on the node
    local_rank = mpi_rank % 8
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if verbose:
        print(f"Connecting to master_addr: {master_addr}")

    # There is a race condition when initializing NCCL with a large number of ranks (e.g 500 ranks)
    # We guard against the failure and then retry
    for attempt_idx in range(n_attempts):
        try:
            dist.init_process_group(backend=backend, init_method=f"env://")
            assert dist.get_rank() == mpi_rank

            use_cuda = torch.cuda.is_available()
            print(f'Using cuda {use_cuda}')
            local_rank = mpi_rank % 8
            device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
            torch.cuda.set_device(local_rank)

            return mpi_rank, local_rank, device
        except RuntimeError as e:
            print(f"Caught error during NCCL init (attempt {attempt_idx} of {n_attempts}): {e}")
            sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd

    raise RuntimeError("Failed to initialize NCCL")
