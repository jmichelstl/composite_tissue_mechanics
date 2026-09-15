/*
 * This file defines the abstract type RandomNetworkBase, which provides a
 * general interface for specifying a randomly generated three dimensional
 * network constrained to lie within a rectangular prismatic bounding box.
 */

#ifndef NETWORK_MESH_COMPLEX
#define NETWORK_MESH_COMPLEX

#include <vector>
#include <tuple>
#include <map>
#include <string>
#include "geometric_primitives.hpp"
#include "random_point_processes.hpp"

using namespace std;
typedef tuple<unsigned, unsigned> edge;
typedef tuple<unsigned, unsigned, unsigned> triple;

//Base type describing a composite fiber network-finiate element mesh (FEM)
//for modeling tissue mechanics
class NetworkMeshComplex {

    public:

        NetworkMeshComplex(){}

        virtual ~NetworkMeshComplex();

        virtual bool build_network(){return true;}

        //Retain each edge with probability p
        void random_dilute(double p);

        //Partition edges into a specified number of segments
        void add_hinges(int num_segments);

        //Use the METIS fill-reducing ordering procedure to permute network
        //network vertices.
        virtual void refine_toplogy();

        virtual void tetrahedralize();

        void set_bounds(Bounds3D new_bounds){
            bounds = new_bounds;
        }

	    void set_length(double new_len){
            mesh_len = new_len;
        }

	    vector<double> get_points(){
            return points;
        }

	    vector<edge> get_edges(){
            return edges;
        }

	    vector<triple> get_triples(){
            return triples;
        }

	    vector<double> get_angles(){
            return angles;
        }

	   vector<int> get_tetrahedra(){
            return tetrahedra;
        }

    protected:
        //Compute triples - series of three consecutive points along a fiber
	    //that participate in a bending interaction.
        virtual void compute_triples(){}

        //Given a set of retained edges following dilution, reassign vertex
        //indices. Cull any triple if one or both of its edges has been culled.
	    void reassign_edges_triples();
	    Bounds3D bounds;
	    double mesh_len;
        vector<double> points;
        vector<edge> edges;
        vector<triple> triples;
	    vector<double> angles;
	    vector<int> tetrahedra;
};

//Type for builing a fiber network-FEM composite using a lattice or lattice
//with a basis to construct the fiber network.
class LatticeBasedNMC : public NetworkMeshComplex {

    public:

        LatticeBasedNMC(){
            bounds = Bounds3D(0, 0, 0, 0, 0, 0);
	        scale = 0;
	        lat_file_name = "";
	        mesh_len = 0;
        }

        LatticeBasedNMC(Bounds3D my_bounds, string lat_file, double my_scale, double my_len){
            bounds = my_bounds;
            lat_file_name = lat_file;
	        scale = my_scale;
	        mesh_len = my_len;
        }

        bool build_network();

        void set_scale(double new_scale){
            scale = new_scale;
        }

        void set_lat_file(string new_name){
            lat_file_name = new_name;
        }

        void tetrahedralize();

        void refine_topology();

    protected:
        //Compute triples by finding pairs of consecutive, parallel line
        //segments.
        void compute_triples();

    private:
        double scale;
        string lat_file_name;
};

//Different point processes from which networks can be generated
enum class point_process {poisson, lloyd};

//Base type for a fiber network-FEM composite in which the fiber network
//is generated from a decoration of a random set of points
class RandomNMCBase : public NetworkMeshComplex {

    public:

        RandomNMCBase(){}

        virtual bool build_network(){return true;}

        void compute_angles();

        void tetrahedralize();

        void random_dilute(double p);

        void refine_topology();

        vector<int> get_tet_ranges(){
            return point_in_tet_ranges;
        }

        vector<int> get_point_in_tet_records(){
            return point_in_tet_records;
        }

        unsigned get_num_net_verts(){
            return num_net_verts;
        }

    protected:
        //Compute triples - series of three points participating in a bending
        //interaction Find the set of edges pairings giving the greatest
	    //geometric mean angle between pairs of outgoing edges from each vertex.
        void compute_triples();
        //The number of vertices in the network, as opposed to the tetrahedral
        //element mesh
        unsigned num_net_verts;
	    //Density with which to seed the volume pervaded by the network with
	    //points
	    double density;
	    //Process for generating random points
        point_process proc;
        //Record of which network vertices are contained in each tetrahedron
        vector<int> point_in_tet_ranges, point_in_tet_records;
};

/*
 * This class generates a randomized variant on the pyrochlore lattice. A
 * hyperuniform point packing is computed, and subsequently the voronoi
 * tessellation of this point packing is determined. The final network is
 * obtained by connecting midpoints of adjacent edges in the voronoi
 * tessellation.
 */
class RandomPyrochloreNMC : public RandomNMCBase {

    public:

        RandomPyrochloreNMC(Bounds3D my_bounds, double my_density, point_process my_proc, double my_len = 0) {
            bounds = my_bounds;
            density = my_density;
            proc = my_proc;
            mesh_len = my_len;
            num_net_verts = 0;
        }

	    bool build_network();
};

#endif
