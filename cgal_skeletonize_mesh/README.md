<h1>MAKE SURE TO COMPILE ON ARCHITECTURE IT WILL RUN ON, OTHERWISE CAN RUN INTO WEIRD RESULTS THAT DON'T SHOW UP AS ERRORS</h1>

To compile this code:
./cgal_create_CMakeLists


Also run `conda install conda-forge::cgal` and `conda install omnia::eigen3`.

Add:
```
# Point at your Conda environment first
list(INSERT CMAKE_PREFIX_PATH 0 "$ENV{CONDA_PREFIX}")
set(CGAL_USE_PKGCONFIG OFF CACHE BOOL "" FORCE)

# Find CGAL
find_package(CGAL 6.0 REQUIRED COMPONENTS Core)
```
and 
```
# Eigen3 integration
find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
include(CGAL_Eigen3_support)   # Defines CGAL::Eigen3_support
```
and
```
# Link in Eigen support
target_link_libraries(skeletonize_mesh
  PRIVATE
    CGAL::Eigen3_support
)
```
to CMakeLists.txt.



cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCGAL_USE_PKGCONFIG=OFF .
make
