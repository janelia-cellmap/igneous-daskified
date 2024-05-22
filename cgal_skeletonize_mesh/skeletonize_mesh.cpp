#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/boost/graph/generators.h>
#include <CGAL/IO/Color.h>

#include <iostream>
#include <fstream>



#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <fstream>

typedef CGAL::Simple_cartesian<double>                        Kernel;
typedef Kernel::Point_3                                       Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::Polyhedron_3<Kernel>                            Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;
//only needed for the display of the skeleton as maximal polylines

//function that takes in Skeleton_vertex and adds it to map if it doesnt exist
int add_vertex_to_map(Skeleton_vertex v, std::map<Skeleton_vertex, int>& vertex_index_map)
{
  if (vertex_index_map.find(v) == vertex_index_map.end())
  {
    vertex_index_map[v] = vertex_index_map.size();
  }
  // return the index of the vertex
  return vertex_index_map[v];
}

// This example extracts a medially centered skeleton from a given mesh.

int main(int argc, char* argv[])
{
  std::string comments;
  // create mesh as epick mesh
  std::ifstream input(argv[1]);
  Polyhedron mesh;
  bool read_ok = CGAL::IO::read_PLY(input, mesh);
  if(!read_ok){
    std::cout << "Error: reading the input file " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  if (!CGAL::is_triangle_mesh(mesh))
  {
    std::cout << "Error: input geometry is not triangulated for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  Skeleton skeleton;
  CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);
  
  // Output all the edges of the skeleton.
  if (boost::num_vertices(skeleton) == 0)
  {
    std::cout << "Error: the skeleton is too small for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  std::ofstream output(argv[2]);

  // create hashmap of Skeleton_vertex to index
  std::map<Skeleton_vertex, int> vertex_index_map;
  for(Skeleton_edge e : CGAL::make_range(edges(skeleton)))
  {
    Skeleton_vertex source_vertex = source(e, skeleton);
    Skeleton_vertex target_vertex = target(e, skeleton);
    int source_id = add_vertex_to_map(source_vertex, vertex_index_map);
    int target_id = add_vertex_to_map(target_vertex, vertex_index_map);

    const Point_3& source_point = skeleton[source(e, skeleton)].point;
    const Point_3& target_point = skeleton[target(e, skeleton)].point;

    output << source_point << " " << target_point << "\n";
  }
  output.close();

  //  Output skeleton points and the corresponding surface points
  output.open("correspondance-poly.polylines.txt");
  for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
    for(vertex_descriptor vd : skeleton[v].vertices)
      output << "2 " << skeleton[v].point << " "
                     << get(CGAL::vertex_point, mesh, vd)  << "\n";

  std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
  std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";


  std::cout << "Success in writing the skeleton to " << argv[2] << std::endl;
  return EXIT_SUCCESS;

}
