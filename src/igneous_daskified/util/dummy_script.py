from dacapo.blockwise.scheduler import run_blockwise
from funlib.geometry import Roi
from igneous_daskified.process.skeletons import get_chunked_skeleton

# Make the ROIs
path_to_worker = "dummy_worker.py"
total_roi = Roi(offset=(0, 0, 0), shape=(100, 100, 100))
read_roi = Roi(offset=(0, 0, 0), shape=(10, 10, 10))
write_roi = Roi(offset=(0, 0, 0), shape=(1, 1, 1))
num_workers = 10

# Run the script blockwise
success = run_blockwise(
    worker_file=path_to_worker,
    total_roi=total_roi,
    read_roi=read_roi,
    write_roi=write_roi,
    num_workers=num_workers,
    # function_name=function_names,
)

# Print the success
if success:
    print("Success")
else:
    print("Failure")

# example run command:
# bsub -n 4 python dummy_script.py
