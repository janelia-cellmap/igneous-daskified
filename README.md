# igneous-daskified

This repository is for creating meshes nad skeletons from segmentations. 

## Installation
Install it via `pip install git+https://github.com/janelia-cellmap/igneous-daskified.git@main
`

### If doing skeletonization
Skeletonization requires building a custom cgal script, located in cgal_skeletonize_mesh. To build it:
1. Clone the repository
2. `cd` to the cgal_skeletonize_mesh directory
3. Run `export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"`
4. Run `cmake -DCMAKE_BUILD_TYPE=Release .`
5. Run `make`

## Types of processing

### Meshes
Meshes can be exported as `ply` or in the [neuroglancer legacy single resolution format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#legacy-single-resolution-mesh-format) or [multi-resolution format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-fragment-data-file-format). Note at present, all meshes - even the multi-resolution ones - are saved in a single resolution.

### Skeletons
Skeletons are saved as polylines in a text file and are also written in the neuroglancer [skeleton format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/skeletons.md).


### Analysis
Mesh analysis can be performed while running the mesh production pipeline, as well as after the fact. Skeleton analysis is by default done during the skeletonization process.

## Command‑Line Usage

The above processes can respectively be run using the commands: `meshify`, `skeletonize`, `analyze-meshes`. They take two parameters:

```bash
usage: [command] [-h] [--num-workers,-n N] config_path
```
`config_path` is the path to the configuration directory which must contain a `run-config.yaml` and a `dask-config.yaml`. The run config specifices the parameters for the particular command to be run, and the dask config parameterizes dask.

Once running, a new directory at `config-path-{DATETIME}` will be created with a copy of the configs used and an `output.log` file for job monitoring and other dask job logs.

Skeletonization of a single mesh can also be run via: `cgal-skeletonize-mesh {input_mesh} {output_file}`.

### Example
An example `run-config.yaml` for `meshify` could be as follows:
```yaml
check_mesh_validity: false
do_analysis: false
do_singleres_multires_neuroglancer: true
downsample_factor: 0
input_path: /path/to/file.zarr/dataset/s0
n_smoothing_iter: 0
output_directory: /path/to/output/dataset
read_write_block_shape_pixels:
- 1250
- 1250
- 1250
remove_smallest_components: false
target_reduction: 0.5
```

A corresponding `dask-config.yaml` for running on an lsf cluster could be:

```yaml
jobqueue:
  lsf:
    # Cluster slots are reserved in chunks.
    # This specifies the chunk size.
    ncpus: 12

    # How many dask worker processed to run per chunk.
    # (Leave one thread empty for garbage collection.)
    processes: 4

    # How many threads to use in each chunk.
    # (Use just one thread per process -- no internal thread pools.)
    cores: 4

    memory: 180GB   # 15 GB per slot
    walltime: 08:00 # Set to 1:00 for access to the short queue
    mem: 180000000000
    use-stdin: true
    log-directory: job-logs
    name: igneous-daskified

    project: cellmap

distributed:
  scheduler:
    work-stealing: true
  admin:
    log-format: '[%(asctime)s] %(levelname)s %(message)s'
    tick:
      interval: 20ms  # time between event loop health checks
      limit: 3h       # time allowed before triggering a warning
```

and for a local config:
```yaml
jobqueue:
  local:
    # Cluster slots are reserved in chunks.
    # This specifies the chunk size.
    ncpus: 1

    # How many dask worker processed to run per chunk.
    # (Leave one thread empty for garbage collection.)
    processes: 1

    # How many threads to use in each chunk.
    # (Use just one thread per process -- no internal thread pools.)
    cores: 1

    log-directory: job-logs
    name: dask-worker

distributed:
  scheduler:
    work-stealing: true
```
If this was a local config, you would run it like: `meshify -n 4 config_path`, if it were on a cluster and you wanted to submit it, you could do something like that `bsub -n 4 meshify -n 4 config_path` where the first `-n` corresponds to the number of workers for the dask scheduler.

The output meshes would be stored in `/path/to/output/dataset/meshes/` and if you want to `do_analysis` (then it is also suggested you set `check_validity: true`), the output metric will be in `/path/to/output/dataset/meshes/metrics/mesh_metrics.csv`.

If running `skeletonize`, txt files with polyline skeletons will be in `/path/to/output/dataset/cgal/`, neuroglancer skeletons will be in `/path/to/output/dataset/skeleton/{full,simplified}/` which will contain - respectively - the neuroglancer formatted full skeletons, and the neuroglancer formatted simplified and pruned skeletons. Corresponding metrics will be in `/path/to/output/dataset/meshes/metrics/skeleton_metrics.csv`