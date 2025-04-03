# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.12...3.29)

project( cgal_skeletonize_mesh )


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()

endif()

find_package(Eigen3 3.4.0)
if (EIGEN3_FOUND)
  include( ${EIGEN3_USE_FILE} )
endif()

# include for local directory

# include for local package


# Creating entries for all C++ files with "main" routine
# ##########################################################


create_single_source_cgal_program( "skeletonize_mesh.cpp" )


