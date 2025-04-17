To compile this code:
./cgal_create_CMakeLists

Add:
```
find_package(Eigen3 3.4.0)
if (EIGEN3_FOUND)
  include( ${EIGEN3_USE_FILE} )
endif()
```

to CMakeLists.txt

cmake -DCMAKE_BUILD_TYPE=Release .
make
