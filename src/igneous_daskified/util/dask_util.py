# %%
from contextlib import contextmanager
import os
import dask
from dask.distributed import Client
import getpass
import tempfile
import shutil
from igneous_daskified.util.io_util import Timing_Messager, print_with_datetime
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
from dataclasses import dataclass
from funlib.persistence import Array
from funlib.geometry import Roi
import numpy as np
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DaskBlock:
    index: int
    roi: Roi


def create_blocks(
    roi: Roi,
    ds: Array,
    read_write_block_shape_pixels=None,
    padding=None,
):
    with Timing_Messager("Generating blocks", logger):
        # roi = roi.snap_to_grid(ds.chunk_shape * ds.voxel_size)
        block_size = read_write_block_shape_pixels
        if read_write_block_shape_pixels is None:
            block_size = ds.chunk_shape
        block_size *= ds.voxel_size
        num_expected_blocks = int(
            np.prod(
                [np.ceil(roi.shape[i] / block_size[i]) for i in range(len(block_size))]
            )
        )
        # create an empty list with num_expected_blocks elements
        block_rois = [None] * num_expected_blocks
        index = 0
        for dim_2 in range(roi.get_begin()[2], roi.get_end()[2], block_size[2]):
            for dim_1 in range(roi.get_begin()[1], roi.get_end()[1], block_size[1]):
                for dim_0 in range(roi.get_begin()[0], roi.get_end()[0], block_size[0]):
                    block_roi = Roi((dim_0, dim_1, dim_2), block_size).intersect(roi)
                    if padding:
                        block_roi = block_roi.grow(padding, padding)
                    block_rois[index] = DaskBlock(index, block_roi.intersect(roi))
                    index += 1
        if index < len(block_rois):
            block_rois[index:] = []
    return block_rois


def set_local_directory(cluster_type):
    """Sets local directory used for dask outputs

    Args:
        cluster_type ('str'): The type of cluster used

    Raises:
        RuntimeError: Error if cannot create directory
    """

    # From https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/dask_util.py
    # This specifies where dask workers will dump cached data

    local_dir = dask.config.get(f"jobqueue.{cluster_type}.local-directory", None)
    if local_dir:
        return

    user = getpass.getuser()
    local_dir = None
    for d in [f"/scratch/{user}", f"/tmp/{user}"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        else:
            local_dir = d
            dask.config.set({f"jobqueue.{cluster_type}.local-directory": local_dir})

            # Set tempdir, too.
            tempfile.tempdir = local_dir

            # Forked processes will use this for tempfile.tempdir
            os.environ["TMPDIR"] = local_dir
            break

    if local_dir is None:
        raise RuntimeError(
            "Could not create a local-directory in any of the standard places."
        )


@contextmanager
def start_dask(num_workers, msg, logger):
    """Context manager used for starting/shutting down dask

    Args:
        num_workers (`int`): Number of dask workers
        msg (`str`): Message for timer
        logger: The logger being used

    Yields:
        client: Dask client
    """

    # Update dask
    with open("dask-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    cluster_type = next(iter(config["jobqueue"]))
    dask.config.update(dask.config.config, config)
    set_local_directory(cluster_type)

    if cluster_type == "local":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(
            n_workers=num_workers, threads_per_worker=1, host="0.0.0.0"
        )
    else:
        if cluster_type == "lsf":
            from dask_jobqueue import LSFCluster

            cluster = LSFCluster()
        elif cluster_type == "slurm":
            from dask_jobqueue import SLURMCluster

            cluster = SLURMCluster()
        elif cluster_type == "sge":
            from dask_jobqueue import SGECluster

            cluster = SGECluster()
        cluster.scale(num_workers)
    try:
        with Timing_Messager(
            f"Starting dask cluster for {msg} with {num_workers} workers", logger
        ):
            client = Client(cluster)
        print_with_datetime(
            f"Check {client.cluster.dashboard_link} for {msg} status.", logger
        )
        yield client
    finally:
        client.shutdown()
        client.close()


def setup_execution_directory(config_path, logger):
    """Sets up the excecution directory which is the config dir appended with
    the date and time.

    Args:
        config_path ('str'): Path to config directory
        logger: Logger being used

    Returns:
        execution_dir ['str']: execution directory
    """

    # Create execution dir (copy of template dir) and make it the CWD
    # from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/bin/launchflow.py
    config_path = config_path[:-1] if config_path[-1] == "/" else config_path
    timestamp = f"{datetime.now():%Y%m%d.%H%M%S}"
    execution_dir = f"{config_path}-{timestamp}"
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(config_path, execution_dir, symlinks=True)
    os.chmod(f"{execution_dir}/run-config.yaml", 0o444)  # read-only
    print_with_datetime(f"Setup working directory as {execution_dir}.", logger)

    return execution_dir
