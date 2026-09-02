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
#include <set>
#include <unordered_set>
#include <suitesparse/cholmod.h>
#include <functional>
#include <tbb.h>
#include <mkl_cblas.h>
#include "geometric_primitives.hpp"
#include "network_mesh_io.hpp"
#include <nlohmann/json.hpp>
#include <matheval.h>
#include <iostream>
#include <assert.h>

using namespace std;
using json = nlohmann::json;

#define BENDING_TOL 1e-12

//Means of classifying the use case - either the network vertices are a proper
//subset of mesh vertices (unified), or the network vertices are separate
//(disjoint)
enum class nmc_type {unified, disjoint};

double compute_i1(double *def_grad);

double compute_i2(double *def_grad);

double compute_j(double *def_grad);

void pk_i1_contribution(double e_derivative, double *def_grad, double *pk_stress);

void pk_i2_contribution(double e_derivative, double *def_grad, double *pk_stress);

void pk_j_contribution(double e_derivative, double *def_grad, double *pk_stress);


//Struct to hold the data needed to describe the model of the gel deformation
//energy and support the computation of forces derived from this energy
struct GelModel {

    GelModel(){
        initialized = false;
        energy_func = NULL;
        arg_names = NULL;
    }

    //Copy constructor to allow duplicates when parallel force and energy
    //calculations are to be performed
    void replicate(GelModel &other){

        char *func_string;
        int length, iter;

        if(! other.initialized){
            initialized = false;
            energy_func = NULL;
            arg_names = NULL;
            return;
        }

        count = other.count;

        //Duplicate the energy function
        func_string = evaluator_get_string(other.energy_func);
        energy_func = evaluator_create(func_string);
        assert(energy_func);

        //Duplicate the list of argument names
        arg_names = (char **) malloc(sizeof(char *) * count);
        for(iter = 0; iter < count; iter ++){
            length = strlen(other.arg_names[iter]) + 1;
            arg_names[iter] = (char *) malloc(sizeof(char) * length);
            strcpy(arg_names[iter], other.arg_names[iter]);
        }

        //Duplicate the list of derivatives w.r.t. principal invariants
        for(iter = 0; iter < other.energy_derivatives.size(); iter ++){
            func_string = evaluator_get_string(other.energy_derivatives[iter]);
            energy_derivatives.push_back(evaluator_create(func_string));
            assert(energy_derivatives[iter]);
        }

        //Duplicate lists of function pointers to compute principal invariants
        //and contributions to the Piola-Kirchoff stress tesnor
        inv_funcs.assign(other.inv_funcs.begin(),other.inv_funcs.end());
        pk_funcs.assign(other.pk_funcs.begin(),other.pk_funcs.end());

        param_values.assign(other.param_values.begin(), other.param_values.end());

        initialized = true;
    }

    bool initialize(json gel_block){

        set<string> invariants, params;
        int iter;
        char **names;

        if(gel_block.contains("free_energy")){
            if(gel_block["free_energy"].is_string()) {
                string expr_string = gel_block["free_energy"].get<string>();
                //Attempt to parse the free energy expression and create a
                //representation of a mathematical equation
                energy_func = evaluator_create((char *) expr_string.c_str());
                if(! energy_func){
                    cerr << "The function could not be parsed.\n";
                    return false;
                }
            }
            else{
                cerr << "The free energy was not specified.\n";
                return false;
            }
        }
        else {
            cerr << "Missing required gel field: free energy\n";
            return false;
        }

        //Enumerate any parameters in the model other than the fundamental
        //invariants, and make sure they are specified
        evaluator_get_variables(energy_func, &names, &count);

        for(iter = 0; iter < count; iter ++){
            if(!strcmp(names[iter], "I1") || !strcmp(names[iter], "I2") || !strcmp(names[iter], "J")){
                invariants.emplace(string(names[iter]));
            }
            else params.emplace(string(names[iter]));
        }

        iter = 0;
        for(string param : params){
            if(gel_block.contains(param)){
                if(gel_block[param].is_number_float()){
                    param_values.push_back(gel_block[param].get<double>());
                }
                else {
                    cerr << "Invalid parameter type\n";
                    evaluator_destroy(energy_func);
                    return false;
                }
            }
            else {
                    cerr << "Missing gel parameter: " << param << endl;
                    evaluator_destroy(energy_func);
                    return false;
            }
            iter ++;
        }

        arg_names = (char **) malloc(sizeof(char *) * count);
        iter = 0;
        for(string next_inv : invariants){
            //Add a record of the invariant to the argument list
            arg_names[iter] = (char *) malloc(sizeof(char)*(next_inv.length()+1));
            strcpy(arg_names[iter], next_inv.c_str());

            //Find the derivative of the free energy density w.r.t. to the
            //current invariant
            energy_derivatives.push_back(evaluator_derivative(energy_func, (char *) next_inv.c_str()));

            //Add function pointers for computing the invariant itself and
            //the associated contribution to the first Piola-Kirchoff stress
            //tensor
            if(next_inv == "I1"){
                inv_funcs.push_back(compute_i1);
                pk_funcs.push_back(pk_i1_contribution);
            }
            else if(next_inv == "I2"){
                inv_funcs.push_back(compute_i2);
                pk_funcs.push_back(pk_i2_contribution);
            }
            else{
                inv_funcs.push_back(compute_j);
                pk_funcs.push_back(pk_j_contribution);
            }

            iter ++;
        }
        for(string next_par : params){
            arg_names[iter] = (char *) malloc(sizeof(char)*(next_par.length()+1));
            strcpy(arg_names[iter], next_par.c_str());
            iter ++;
        }

        initialized = true;
        return initialized;
    }

    //Free memory allocated in initializing gel model fields
    void clear(){

        int iter;

        if(! initialized) return;

        evaluator_destroy(energy_func);
        for(iter = 0; iter < energy_derivatives.size(); iter ++){
            evaluator_destroy(energy_derivatives[iter]);
        }

        for(iter = 0; iter < count; iter ++){
            free(arg_names[iter]);
        }
        free(arg_names);

        param_values.clear();

        initialized = false;
        energy_func = NULL;
        arg_names = NULL;
    }

    //Argument count
    int count;

    //Record of whether or not the structure has been successfully
    //initialized
    bool initialized;

    //Free energy density
    void *energy_func;
    
    //Derivatives of the energy density w.r.t. principle invariants
    vector<void *> energy_derivatives;

    //Function pointers to compute each required principle invariant
    vector<double(*)(double *def_grad)> inv_funcs;

    //Function pointers to compute components of the first Piola-Kirchoff stress
    //tensor
    vector<void(*)(double e_deriv, double *def_grad, double *pk_stress)> pk_funcs;

    //Names of arguments to the energy density function and its derivatives
    //and model parameter values
    char **arg_names;
    vector<double> param_values;
};

//Data structre to hold information pertaining structural relaxation
struct NetworkMeshData {

    NetworkMeshData(){
        invariants = NULL;
        def_grads = NULL;
        pk_tensors = NULL;
    }

    NetworkMeshData(nmc_type my_type) : type(my_type) {
        invariants = NULL;
        def_grads = NULL;
        pk_tensors = NULL;
    }

    //If the network is not lattice based and has bond of varying length,
    //different stretching and bending constants will be needed, in general,
    //for different edges and hinges. Edges and triples should be formed
    //and edge lengths computed before these functions are invoked.
    void compute_stretching_constants(){

        for(int iter = 0; iter < edges.size(); iter ++){
            stretching_constants.push_back(ks / lengths[iter]);
        }
    }

    void compute_bending_constants(){
        int trip_iter, idx1, idx2, idx3;
        Vec3D p1, p2, p3;

        for(trip_iter = 0; trip_iter < triples.size(); trip_iter += 3){
            assign(p1, points, triples[trip_iter]);
            assign(p2, points, triples[trip_iter+1]);
            assign(p3, points, triples[trip_iter+2]);
            bending_constants.push_back(2*kb/(distance(p1,p2)+distance(p2,p3)));
        }
    }

    //Fields commmon to all use cases

    //Geometrical and topological data
    vector<double> bounds, points, lengths, angles, inv_mats, volumes;
    vector<int> edges, triples, tetrahedra;

    //Constrained points that should not be allowed to relax
    unordered_set<int> bottom, top;
    nmc_type type;

    //Information for calculating the deformation energy and forces
    double ks, kb;
    GelModel gmod;

    //This field is only used when parallel force and energy calculations
    //are performed, and is needed in such cases because the libmatheval
    //library is not thread safe.
    vector<GelModel> model_copies;

    //These fields are used to divide work among threads when parallel
    //force and energy force calculations are desired.
    vector<int> stretch_assignments, bend_assignments, gel_assignments;
    //Principal scalar invariants, deformation gradient tensors, and First
    //Piola-Kirchoff stress tensors, used to expose more parallelism.
    //vector<double> invariants, def_grads, pk_tensors;
    double *invariants, *def_grads, *pk_tensors;

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

void gel_forces(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, GelModel &gmod, vector<double> &forces, int start, int end);

void gel_forces_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, GelModel &gmod, vector<double> &forces, int start, int end);

double calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, GelModel gmod);

double calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, GelModel gmod);

//double parallel_calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<GelModel> gmods, vector<int> gel_assignments);
double parallel_calc_gel_energy(NetworkMeshData &nmd);

//double parallel_calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, vector<GelModel> gmods, vector<int> gel_assignments);
double parallel_calc_gel_energy_disjoint(NetworkMeshData &nmd);

void compute_forces_unified(NetworkMeshData &nmd, vector<double> &forces);

void compute_forces_disjoint(NetworkMeshData &nmd, vector<double> &forces);

//void compute_forces_unified_parallel(NetworkMeshData &nmd, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);
void compute_forces_unified_parallel(NetworkMeshData &nmd, vector<double> &forces, vector<vector<double>> &force_sets);

//void compute_forces_disjoint_parallel(NetworkMeshData &nmd, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);
void compute_forces_disjoint_parallel(NetworkMeshData &nmd, vector<double> &forces, vector<vector<double>> &force_sets);

#endif
