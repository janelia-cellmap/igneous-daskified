import os
from funlib.geometry import Roi, Coordinate
from igneous_daskified.util import dask_util, io_util
from funlib.persistence import open_ds
import logging
import sys
import importlib
import os
from dataclasses import dataclass

############################################
# HACKY WAY TO IMPORT PYMESHLAB CORRECTLY
spec = importlib.util.find_spec("pymeshlab")
pymesh_lib_path = spec.origin.rsplit("/", 1)[0] + "/lib"

if os.getenv("LD_LIBRARY_PATH") == None:
    os.environ["LD_LIBRARY_PATH"] = pymesh_lib_path
    try:
        sys.stdout.flush()
        os.execl(sys.executable, sys.executable, *sys.argv)
    except OSError as e:
        print(e)
elif pymesh_lib_path not in os.getenv("LD_LIBRARY_PATH"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [pymesh_lib_path, os.getenv("LD_LIBRARY_PATH")]
    )
    try:
        sys.stdout.flush()
        os.execl(sys.executable, sys.executable, *sys.argv)
    except OSError as e:
        print(e)
############################################

logger = logging.getLogger(__name__)


@dataclass
class RunProperties:
    def __init__(self):
        args = io_util.parser_params()

        # Change execution directory
        self.execution_directory = dask_util.setup_execution_directory(
            args.config_path, logger
        )
        self.logpath = f"{self.execution_directory}/output.log"
        self.run_config = io_util.read_run_config(args.config_path)
        self.run_config["num_workers"] = args.num_workers


def skeletonize():
    from .skeletons import Skeletonize

    rp = RunProperties()
    # Start mesh creation
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        skeletonize = Skeletonize(**rp.run_config)
        skeletonize.get_skeletons()


def meshify():
    from .meshes import Meshify

    rp = RunProperties()
    # Start mesh creation
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        meshify = Meshify(**rp.run_config)
        meshify.get_meshes()


def analyze_meshes():
    from .analyze import AnalyzeMeshes

    rp = RunProperties()
    # Start analsis
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        analyze_meshes = AnalyzeMeshes(**rp.run_config)
        analyze_meshes.analyze()
