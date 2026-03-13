<h1>MAKE SURE TO COMPILE ON ARCHITECTURE IT WILL RUN ON, OTHERWISE CAN RUN INTO WEIRD RESULTS THAT DON'T SHOW UP AS ERRORS</h1>

To compile this code:
./cgal_create_CMakeLists


Also run `conda install conda-forge::cgal` and `conda install omnia::eigen3`.


cd /path/to/cgal_skeletonize_mesh
rm -f CMakeCache.txt
rm -rf CMakeFiles
rm -f Makefile cmake_install.cmake
cmake -S . -B . -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
make -j$(nproc)
