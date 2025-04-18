# %%
import os
import sys
import time
import socket
import smtplib
import getpass
import logging
import traceback
from email.mime.text import MIMEText
from contextlib import ContextDecorator, contextmanager
from subprocess import Popen, PIPE, TimeoutExpired, run as subprocess_run
from datetime import datetime
import argparse
import yaml
from yaml.loader import SafeLoader
from funlib.geometry import Roi
from funlib.persistence import open_ds
import numpy as np
import io
import json
import DracoPy
import struct
from cloudvolume.datasource.precomputed.mesh.multilod import (
    MultiLevelPrecomputedMeshManifest,
    to_stored_model_space,
)
import fastremap


# write_ngmesh taken from vol2mesh https://github.com/janelia-flyem/vol2mesh/blob/1b667bcd45423bfb7ea17c42a135931c6decf752/vol2mesh/mesh.py#L1032
def _write_ngmesh(vertices_xyz, faces, f_out):
    """
    Write the given vertices (verbatim) and faces to the given
    binary file object, which must already be open.
    """
    f_out.write(np.uint32(len(vertices_xyz)))
    f_out.write(vertices_xyz.astype(np.float32, "C", copy=False))
    f_out.write(faces.astype(np.uint32, "C", copy=False))


def write_ngmesh(vertices_xyz, faces, f_out=None):
    """
    Write the given vertices and faces to the given output path/file,
    in ngmesh format as described above.

    Args:
        vertices_xyz:
            vertex array, shape (V,3)

        faces:
            face index array, shape (F,3), referring to the rows of vertices_xyz

        f_out:
            If None, bytes are returned
            If a file path, the data is written to a file at that path.
            Otherwise, must be an open binary file object.
    Returns:
        If f_out is None, bytes are returned.
        Otherwise the mesh is written to f_out and None is returned.
    """
    if f_out is None:
        # Return as bytes
        with io.BytesIO() as bio:
            _write_ngmesh(vertices_xyz, faces, bio)
            return bio.getvalue()

    elif isinstance(f_out, str):
        # Write to a path
        with open(f_out, "wb") as f:
            _write_ngmesh(vertices_xyz, faces, f)

    else:
        # Write to the given file object
        _write_ngmesh(vertices_xyz, faces, f)


def write_ngmesh_metadata(meshdir):
    mesh_ids = [f.split(":0")[0] for f in os.listdir(meshdir) if ":0" in f]
    info = {
        "@type": "neuroglancer_legacy_mesh",
        "segment_properties": "./segment_properties",
    }

    with open(meshdir + "/info", "w") as f:
        f.write(json.dumps(info))

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [mesh_id for mesh_id in mesh_ids],
            "properties": [
                {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
            ],
        },
    }
    os.makedirs(meshdir + "/segment_properties", exist_ok=True)
    with open(meshdir + "/segment_properties/info", "w") as f:
        f.write(json.dumps(segment_properties))


def write_index_file(
    path,
    grid_origin,
    fragment_positions,
    fragment_offsets,
    current_lod,
    lods,
    chunk_shape,
):
    """Write the index files for a mesh.

    Args:
        path: Path to mesh
        grid_origin: The lod 0 mesh grid origin
        fragments: Fragments for current lod
        current_lod: The current lod
        lods: A list of all the lods
        chunk_shape: Chunk shape.
    """

    # since we don't know if the lowest res ones will have meshes for all svs
    lods = [lod for lod in lods if lod <= current_lod]

    num_lods = len(lods)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0.0, 0.0, 0.0] for _ in range(num_lods)])
    num_fragments_per_lod = np.array([len(fragment_positions)])

    # if current_lod == lods[0] or not os.path.exists(f"{path}.index"):
    # then is highest res lod or if the file doesnt exist yet it failed
    # to write out the index file because s0 was draco compressed to nothing
    # in encode_faces_to_custom_drc_bytes due to voxel size and chunk shape
    # build a list of raw byte‐sequences
    blocks = [
        chunk_shape.astype("<f").tobytes(),
        grid_origin.astype("<f").tobytes(),
        struct.pack("<I", num_lods),
        lod_scales.astype("<f").tobytes(),
        vertex_offsets.astype("<f").tobytes(order="C"),
        num_fragments_per_lod.astype("<I").tobytes(),
        # fragment positions and offsets
        np.asarray([fragment_position for fragment_position in fragment_positions])
        .T.astype("<I")
        .tobytes(order="C"),
        np.asarray([fragment_offset for fragment_offset in fragment_offsets])
        .astype("<I")
        .tobytes(order="C"),
    ]
    with open(f"{path}", "wb") as f:
        f.writelines(blocks)
    # else:
    #    rewrite_index_with_empty_fragments(path, fragments)


def to_stored_model_space(
    vertices: np.ndarray,
    chunk_shape,
    grid_origin,
    fragment_positions,
    vertex_offsets,
    lod: int,
    vertex_quantization_bits: int,
) -> np.ndarray:
    """Inverse of from_stored_model_space (see explaination there)."""
    vertices = vertices.astype(np.float32, copy=False)
    quant_factor = (2**vertex_quantization_bits) - 1

    stored_model = vertices - grid_origin - vertex_offsets
    stored_model /= chunk_shape * (2**lod)
    stored_model -= fragment_positions
    stored_model *= quant_factor
    stored_model = np.round(stored_model, out=stored_model)
    stored_model = np.clip(stored_model, 0, quant_factor, out=stored_model)

    dtype = fastremap.fit_dtype(np.uint64, value=quant_factor)
    return stored_model.astype(dtype)


def write_singleres_multires_files(
    vertices_xyz, faces, path, vertex_quantization_bits=10, draco_compression_level=10
):
    grid_origin = np.min(vertices_xyz, axis=0)
    chunk_shape = np.max(vertices_xyz, axis=0) - grid_origin
    # NOTE: should use cloudvolume one
    vertices_xyz = to_stored_model_space(
        vertices_xyz,
        chunk_shape=chunk_shape,
        grid_origin=grid_origin,
        fragment_positions=np.array([[0, 0, 0]]),
        vertex_offsets=np.array([0, 0, 0]),
        lod=0,
        vertex_quantization_bits=vertex_quantization_bits,
    )

    quantization_origin = np.min(vertices_xyz, axis=0)
    quantization_range = np.max(vertices_xyz, axis=0) - quantization_origin
    quantization_range = np.max(quantization_range)
    try:
        res = DracoPy.encode(
            vertices_xyz,
            faces,
            quantization_bits=vertex_quantization_bits,
            quantization_range=quantization_range,
            compression_level=draco_compression_level,
            quantization_origin=quantization_origin,
        )
    except DracoPy.EncodingFailedException:
        res = b""

    # write res to binary file
    with open(path, "wb") as f:
        f.write(res)

    write_index_file(
        f"{path}.index",
        grid_origin=grid_origin,
        fragment_positions=[[0, 0, 0]],
        fragment_offsets=[len(res)],
        current_lod=0,
        lods=[0],
        chunk_shape=chunk_shape,
    )
    return res, vertices_xyz


def write_singleres_multires_metadata(meshdir):
    mesh_ids = [f.split(".index")[0] for f in os.listdir(meshdir) if ".index" in f]
    with open(f"{meshdir}/info", "w") as f:
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 10,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 1,
            "segment_properties": "segment_properties",
        }

        json.dump(info, f)

    with open(meshdir + "/info", "w") as f:
        f.write(json.dumps(info))

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [mesh_id for mesh_id in mesh_ids],
            "properties": [
                {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
            ],
        },
    }
    os.makedirs(meshdir + "/segment_properties", exist_ok=True)
    with open(meshdir + "/segment_properties/info", "w") as f:
        f.write(json.dumps(segment_properties))


def write_multires_info_file(path):
    """Write info file for meshes

    Args:
        path ('str'): Path to meshes
    """
    # default to 10 quantization bits
    with open(f"{path}/info", "w") as f:
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 10,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 1,
            "segment_properties": "segment_properties",
        }

        json.dump(info, f)


# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class Timing_Messager(ContextDecorator):
    """Context manager class to time operations"""

    def __init__(self, base_message, logger):
        """Initialize instance with a message and logger

        Args:
            base_message ('str'): Message for logger
            logger: logger to be used
        """

        self._base_message = base_message
        self._logger = logger

    def __enter__(self):
        """Set the start time and print the status message"""

        print_with_datetime(f"{self._base_message}...", self._logger)
        self._start_time = time.time()
        return self

    def __exit__(self, *exc):
        """Print the exit message and elapsed time"""

        print_with_datetime(
            f"{self._base_message} completed in {time.time()-self._start_time}!",
            self._logger,
        )
        return False


def print_with_datetime(output, logger):
    """[summary]

    Args:
        output ([type]): [description]
        logger ([type]): [description]
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"{now}: {output}")


def read_run_config(config_path):
    """Reads the run config from config_path and stores them

    Args:
        config_path ('str'): Path to config directory

    Returns:
        Dicts of required_settings and optional_decimation_settings
    """

    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def parser_params():
    """Parse command line parameters including the config path and number of workers."""

    parser = argparse.ArgumentParser(
        description="Code to convert single-scale (or a set of multi-scale) meshes to the neuroglancer multi-resolution mesh format"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to directory containing run-config.yaml and dask-config.yaml",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Number of workers to launch (i.e. each worker is launched with a single bsub command)",
    )

    return parser.parse_args()


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.

    Redirects a file object or file descriptor to a new file descriptor.

    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')

    Motivation: In pure-Python, you can redirect all print() statements like this:

        sys.stdout = open('myfile.txt')

        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.
    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.

    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError:  # filename
        pass

    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        # libc.fflush(None)  # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def tee_streams(output_path, append=False):
    """
    Context manager.
    All stdout and stderr will be tee'd to a a file on disk.
    (in addition to appearing on the original stdout streams).

    Note: Stdout and stderr will be merged, in both the tee file and the console.
    """
    if append:
        append = "-a"
    else:
        append = ""

    tee = Popen(
        f"tee {append} {output_path}",
        shell=True,
        stdin=PIPE,
        bufsize=1,
        universal_newlines=True,  # line buffering
        preexec_fn=os.setpgrp,
    )  # Spawn the tee process in its own process group,
    # so it won't receive SIGINT.
    # (Otherwise it might close its input stream too early if the user hits Ctrl+C.)
    try:
        try:
            with stdout_redirected(tee.stdin, stdout=sys.stdout):  # pipe stdout to tee
                with stdout_redirected(
                    sys.stdout, stdout=sys.stderr
                ):  # merge stderr into stdout
                    yield
        finally:
            tee.stdin.close()
            try:
                tee.wait(1.0)
            except TimeoutExpired:
                pass
    except:
        # If an exception was raised, append the traceback to the file
        with open(output_path, "a") as f:
            traceback.print_exc(file=f)
        raise


@contextmanager
def email_on_exit(email_config, workflow_name, execution_dir, logpath):
    """
    Context manager.

    Sends an email when the context exits with success/fail status in the subject line.
    Doesn't work unless sendmail() works without a password
    (i.e. probably won't work on your laptop, will work on a Janelia cluster node).

    Args:
        email_config:
            See flyemflows.workflow.base.base_schemas.ExitEmailSchema

        workflow_name:
            Name of the workflow class to be reported in the email.

        execution_dir:
            Location of the workflow config/data files to be reported in the email.

        logpath:
            Location of the logfile whose contents will be included in
            the email if email_config["include-log"] is True.

    """
    if not email_config["send"]:
        yield
        return

    if not email_config["addresses"]:
        logger.warning(
            "Your config enabled the exit-email feature, but "
            "no email addresses were listed. Nothing will be sent."
        )
        yield
        return

    user = getpass.getuser()
    host = socket.gethostname()
    jobname = os.environ.get("LSB_JOBNAME", None)

    addresses = []
    for address in email_config["addresses"]:
        if address == "JANELIA_USER":
            address = f"{user}@janelia.hhmi.org"
        addresses.append(address)

    with Timer() as timer:

        def send_email(headline, result, error_str=None):
            body = (
                headline + f"Duration: {timer.timedelta}\n"
                f"Execution directory: {execution_dir}\n"
            )

            if jobname:
                body += f"Job name: {jobname}\n"

            if error_str:
                body += f"Error: {error_str}\n"

            if email_config["include-log"]:
                # Sync first, in the hope that the log will flush to disk before we read it.
                # Note:
                #    Currently raised exceptions haven't been printed yet,
                #    so they aren't yet in the log file in your email.
                #    They'll only be present in the on-disk logfile.
                try:
                    # This can hang, apparently.
                    # Hangs like this might be fairly damaging, unfortunately.
                    # According to Ken:
                    #   >If sync is trying to write a file down to disk that was deleted,
                    #   >it can hang like that. Unfortunately, the node will have to be
                    #   >power cycled to deal with this situation.
                    #
                    # Let's hope that's not common.
                    # We'll just timeout the ordinary way and hope for the best.
                    subprocess_run("sync", timeout=10.0)
                    time.sleep(2.0)
                except TimeoutExpired:
                    logger.warning("Timed out while waiting for filesystem sync")

                body += "\nLOG (possibly truncated):\n\n"
                with open(f"{logpath}", "r") as log:
                    body += log.read()

            msg = MIMEText(body)
            msg["Subject"] = f"Workflow exited: {result}"
            msg["From"] = f"flyemflows <{user}@{host}>"
            msg["To"] = ",".join(addresses)

            try:
                s = smtplib.SMTP("mail.hhmi.org")
                s.sendmail(msg["From"], addresses, msg.as_string())
                s.quit()
            except:
                msg = (
                    "Failed to send completion email.  Perhaps your machine "
                    "is not configured to send login-less email, which is required for this feature."
                )
                logger.error(msg)

        try:
            yield
        except BaseException as ex:
            send_email(
                f"Workflow {workflow_name} failed: {type(ex)}\n", "FAILED", str(ex)
            )
            raise
        else:
            send_email(f"Workflow {workflow_name} exited successfully.\n", "success")


# import trimesh

# mesh = trimesh.load_mesh(
#     "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/all/tmp_chunked/37/block_0.ply"
# )
# res, vertices_xyz = write_singleres_multires_files(
#     mesh.vertices,
#     mesh.faces,
#     "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/test_multires/37",
# )

# # res, quantization_origin, quantization_range = _write_multires_mesh(
# #     mesh.vertices, mesh.faces
# # )


# %%
# import struct
# import numpy as np


# def read_index(path):
#     """
#     Reads a “.index” file with the format:

#       chunk_shape:      3×float32le
#       grid_origin:      3×float32le
#       num_lods:         uint32le
#       lod_scales:       num_lods×float32le
#       vertex_offsets:   (num_lods×3)×float32le, reshape→(num_lods,3)
#       num_fragments_per_lod: num_lods×uint32le
#       then, for each lod in [0..num_lods):
#         fragment_positions:  (3×N)×uint32le, reshape→(3, N)
#         fragment_offsets:    N×uint32le

#     Returns:
#       chunk_shape, grid_origin : (3,) float32 arrays
#       num_lods                 : int
#       lod_scales               : (num_lods,) float32 array
#       vertex_offsets           : (num_lods,3) float32 array
#       num_fragments_per_lod    : (num_lods,) uint32 array
#       fragment_positions       : list of (3, N_lod) uint32 arrays
#       fragment_offsets         : list of (N_lod,) uint32 arrays
#     """
#     with open(f"{path}.index", "rb") as f:
#         # read fixed‐size headers
#         chunk_shape = np.frombuffer(f.read(3 * 4), dtype="<f4")  # 3 floats
#         grid_origin = np.frombuffer(f.read(3 * 4), dtype="<f4")  # 3 floats

#         num_lods_bytes = f.read(4)
#         num_lods = struct.unpack("<I", num_lods_bytes)[0]  # uint32

#         lod_scales = np.frombuffer(f.read(num_lods * 4), dtype="<f4")  # num_lods floats
#         print(chunk_shape, grid_origin, num_lods, lod_scales)
#         vertex_offsets = np.frombuffer(
#             f.read(num_lods * 3 * 4), dtype="<f4"
#         ).reshape(  # num_lods×3 floats
#             (num_lods, 3), order="C"
#         )

#         num_fragments_per_lod = np.frombuffer(
#             f.read(num_lods * 4), dtype="<u4"
#         )  # num_lods uint32s

#         # now read each LOD’s fragment data
#         fragment_positions = []
#         fragment_offsets = []

#         for lod in range(num_lods):
#             n = int(num_fragments_per_lod[lod])

#             # positions: 3×n uint32s, stored in C order as [3, n]
#             pos = np.frombuffer(f.read(n * 3 * 4), dtype="<u4").reshape(
#                 (3, n), order="C"
#             )
#             fragment_positions.append(pos)

#             # offsets: n uint32s
#             off = np.frombuffer(f.read(n * 4), dtype="<u4")
#             fragment_offsets.append(off)

#     return (
#         chunk_shape,
#         grid_origin,
#         num_lods,
#         lod_scales,
#         vertex_offsets,
#         num_fragments_per_lod,
#         fragment_positions,
#         fragment_offsets,
#     )


# read_index(
#     "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/all/meshes/37"
# )
# # %%
# import struct
# import numpy as np


# def write_index(
#     path: str,
#     chunk_shape: np.ndarray,
#     grid_origin: np.ndarray,
#     lod_scales: np.ndarray,
#     vertex_offsets: np.ndarray,
#     num_fragments_per_lod: np.ndarray,
#     fragment_positions: list,
#     fragment_offsets: list,
# ):
#     """
#     Writes out a “.index” file with exactly the same binary layout described
#     in read_index().
#     """
#     # enforce dtype + shape
#     cs = np.asarray(chunk_shape, dtype="<f4").reshape(3)
#     go = np.asarray(grid_origin, dtype="<f4").reshape(3)

#     ls = np.asarray(lod_scales, dtype="<f4").ravel()
#     num_lods = ls.size

#     vo = np.asarray(vertex_offsets, dtype="<f4").reshape((num_lods, 3), order="C")
#     nfr = np.asarray(num_fragments_per_lod, dtype="<u4").ravel()
#     if nfr.size != num_lods:
#         raise ValueError("num_fragments_per_lod must have length num_lods")

#     # build blocks in order
#     blocks = [
#         cs.tobytes(order="C"),
#         go.tobytes(order="C"),
#         struct.pack("<I", num_lods),
#         ls.tobytes(order="C"),
#         vo.tobytes(order="C"),
#         nfr.tobytes(order="C"),
#     ]

#     # per‐LOD data
#     for lod in range(num_lods):
#         count = int(nfr[lod])

#         pos = np.asarray(fragment_positions[lod], dtype="<u4")
#         if pos.shape != (3, count):
#             raise ValueError(f"fragment_positions[{lod}] must be (3, {count})")
#         blocks.append(pos.tobytes(order="C"))

#         off = np.asarray(fragment_offsets[lod], dtype="<u4").ravel()
#         if off.size != count:
#             raise ValueError(f"fragment_offsets[{lod}] length must be {count}")
#         blocks.append(off.tobytes(order="C"))

#     # write everything in one go
#     with open(f"{path}.index", "wb") as f:
#         f.writelines(blocks)

# %%
