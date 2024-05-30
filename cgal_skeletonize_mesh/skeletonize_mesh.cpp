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
struct Display_polylines{
  const Skeleton& skeleton;
  std::ofstream& out;
  int polyline_size;
  std::stringstream sstr;
  Display_polylines(const Skeleton& skeleton, std::ofstream& out)
    : skeleton(skeleton), out(out)
  {}
  void start_new_polyline(){
    polyline_size=0;
    sstr.str("");
    sstr.clear();
  }
  void add_node(Skeleton_vertex v){
    ++polyline_size;
    sstr << " " << v;//skeleton[v].point;
  }
  void end_polyline()
  {
    out << "p " << sstr.str() << "\n";
  }
};

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


  //  Output skeleton points and the corresponding surface points
  std::ofstream output(argv[2]);
  // assign indices to vertices
  for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton))){
    Point_3 skeleton_vertex = skeleton[v].point;
    float average_radius, radius = 0;
    if (skeleton[v].vertices.size() == 0)
      {// some vertices aren't associated with mesh vertices but we still want to calculate a radius for them
        float min_radius = std::numeric_limits<float>::max();
        //loop over all vertices of mesh and find the minimum distance between the skeleton vertex and the mesh vertices
        for(vertex_descriptor mesh_vd : vertices(mesh)){
          Point_3 mesh_vertex = get(CGAL::vertex_point, mesh, mesh_vd);
          radius = std::sqrt(CGAL::squared_distance(mesh_vertex, skeleton_vertex));
          if(radius < min_radius)
            min_radius = radius;
        }
        average_radius = min_radius;
      }
      else{
        // get the average associated vertex distance
        for(vertex_descriptor vd : skeleton[v].vertices){
          Point_3 associated_mesh_vertex = get(CGAL::vertex_point, mesh, vd);
          average_radius += std::sqrt(CGAL::squared_distance(associated_mesh_vertex, skeleton_vertex));
        }
        average_radius /= skeleton[v].vertices.size();
      }
      output << "v " << skeleton[v].point << " " << average_radius << "\n" ;
  }

  for(Skeleton_edge e : CGAL::make_range(edges(skeleton)))
  {
    Skeleton_vertex source_vertex = source(e, skeleton);
    Skeleton_vertex target_vertex = target(e, skeleton);

    output << "e " << source_vertex << " " << target_vertex << "\n";

  }


  //output.close();

  // Output all the edges of the skeleton.
  //std::ofstream output("skel-poly.polylines.txt");
  Display_polylines display(skeleton,output);
  CGAL::split_graph_into_polylines(skeleton, display);
  output.close();

  // //  Output skeleton points and the corresponding surface points
  // output.open("correspondance-poly.polylines.txt");
  // for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
  //   for(vertex_descriptor vd : skeleton[v].vertices)
  //     output << "2 " << skeleton[v].point << " "
  //                    << get(CGAL::vertex_point, mesh, vd)  << "\n";

  // std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
  // std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";


  std::cout << "Success in writing the skeleton to " << argv[2] << std::endl;
  return EXIT_SUCCESS;

}
