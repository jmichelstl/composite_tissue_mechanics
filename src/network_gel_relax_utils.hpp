/*
 * This library provides common functionality for evaluating the energies and
 * forces associated with the deformation of a filamentous network and a gel
 * modeled by a tetrahedral finite element mesh, and for performing structural
 * relaxation of such systems. Two situations are addressed: that in which the
 * network vertices are a proper subset of the tetrahedron vertices, and that in
 * which the network vertices and tetrahedron vertices belong to disjoint sets.
 */

#ifndef NGRU
#define NGRU

#include <vector>
#include <unordered_set>
#include <suitesparse/cholmod.h>
#include <functional>
#include <tbb.h>
#include <mkl_cblas.h>
#include "geometric_primitives.hpp"
#include "network_mesh_io.hpp"

using namespace std;

#define BENDING_TOL 1e-12

//Means of classifying the use case - either the network vertices are a proper
//subset of mesh vertices (unified), or the network vertices are separate
//(disjoint)
enum class nmc_type {unified, disjoint};

//Data structre to hold information pertaining structural relaxation
struct NetworkMeshData {

    NetworkMeshData(){}

    NetworkMeshData(nmc_type my_type) : type(my_type) {}

    //Fields commmon to all use cases
    vector<double> bounds, points, lengths, angles, inv_mats, volumes;
    vector<int> edges, triples, tetrahedra;
    unordered_set<int> bottom, top;
    nmc_type type;

    //Fields needed for disjoint network-mesh complexes
    //These are ignored if the network vertices are a subset of the mesh
    //vertices.
    
    //Offset to beginning of network vertices
    unsigned num_net_verts;
    //Information about which network vertices are contained in each tetrahedral
    //mesh element
    vector<int> vertex_ranges, contained_vertices;
    //Reference locations of network vertices, which are used in computing
    //displaced locations using corresponding mesh elements' local affine
    //transformations, and stretching and bending constants, which can
    //vary because of the variable lengths of edges in off-lattice networks.
    vector<double> reference_points, stretching_constants, bending_constants; 
    //Matrix to map network forces onto mesh vertex forces, as well as a
    //SuiteSparse context to manage sparse matrix operations
    cholmod_common common;
    cholmod_sparse *force_mapping;
};

//Function to extract the contents of a network-mesh complex and obtain 
//information needed for structural relaxation calculations. 
bool parse_nmc_file(string file_name, NetworkMeshData &nmd);

//Function to support a computation of a map from network forces to
//contributions to forces on mesh vertices
bool compute_network_mesh_map(NetworkMeshData &nmd);

//Utility function for the parallel summation of arrays
template <typename T> T sum_reduce(const T* first, const T* last, T identity);

//Utility function for partitioning jobs accross threads
vector<int> job_assignments(int job_num, int thread_num);

//Load coordinates into a vector
void load_coords(vector<double> points, int idx, Vec3D &pt);

//Compute the volumes of tetrahedra
void calc_volumes(NetworkMeshData &nmd);

void get_inv_mats(vector<double> points, vector<int> tets, vector<double> &inv_mats, nmc_type type);

void stretching_forces(vector<double> points, vector<int> edges, vector<double> lengths, double k, vector<double> &forces, int start, int end);

void stretching_forces_disjoint(vector<double> points, vector<int> edges, vector<double> lengths, vector<double> constants, vector<double> &forces, int start, int end);

double stretching_energy(vector<double> points, vector<int> edges, vector<double> lengths, double k);

double stretching_energy_disjoint(vector<double> points, vector<int> edges, vector<double> lengths, vector<double> constants);

void stretching_energy_worker(vector<double> points, vector<int> edges, vector<double> lengths, double k, int index, double *energy);

void stretching_energy_worker_disjoint(vector<double> points, vector<int> edges, vector<double> lengths, vector<double> constants, int index, double *energy);

double parallel_stretching_energy(vector<double> points, vector<int> edges, vector<double> lengths, double ks);

double parallel_stretching_energy_disjoint(vector<double> points, vector<int> edges, vector<double> lengths, vector<double> constants);

void bending_forces(vector<double> pts, vector<int> triples, vector<double> angles, double kb, vector<double> &forces, int start, int end);

void bending_forces_disjoint(vector<double> pts, vector<int> triples, vector<double> angles, vector<double> constants, vector<double> &forces, int start, int end);

double fiber_bending_energy(vector<double> points, vector<int> triples, vector<double> angles, double kb);

double fiber_bending_energy_disjoint(vector<double> points, vector<int> triples, vector<double> angles, vector<double> constants);

double parallel_fiber_bending_energy(vector<double> points, vector<int> triples,vector<double> angles, double kb);

double parallel_fiber_bending_energy_disjoint(vector<double> points, vector<int> triples,vector<double> angles, vector<double> constants);

void gel_forces(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi, vector<double> &forces, int start, int end);

void gel_forces_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi, vector<double> &forces, int start, int end);

double calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi);

double calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi);

double parallel_calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi);

double parallel_calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi);

void compute_forces_unified(NetworkMeshData &nmd, vector<double> params, vector<double> &forces);

void compute_forces_disjoint(NetworkMeshData &nmd, vector<double> params, vector<double> &forces);

void compute_forces_unified_parallel(NetworkMeshData &nmd, vector<double> params, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);

void compute_forces_disjoint_parallel(NetworkMeshData &nmd, vector<double> mech_params, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);

void compute_stretching_constants(NetworkMeshData &nmd, double ks);

void compute_bending_constants(NetworkMeshData &nmd, double kb);

#endif
