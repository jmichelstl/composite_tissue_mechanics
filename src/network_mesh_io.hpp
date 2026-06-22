#include <vector>
#include <stdio.h> 
#include <iostream> 
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <zlib.h>
#include <assert.h>
#include "geometric_primitives.hpp"
#include "network_mesh_complex.hpp"
#include <tinyxml2.h>
#include <memory>

using namespace std;

//Functionality for writing and reading data related to relaxation runs

//Expected number of parameters
#define EXPECTED_PARAM_COUNT 6

//Type to represent the physical parameters for relaxation run
struct RelaxParams {

    RelaxParams(){}

    RelaxParams(double ks, double kb, double gm, double eo, double ut, double my_chi){
        stretch = ks;
	    bend = kb;
        gel_mod = gm;
        excess_osm = eo;
        u_thermal = ut;
        chi = my_chi;
    }

    double stretch, bend, gel_mod, excess_osm, u_thermal, chi;
};

//Type to represent one of a series of steps in a relaxation run, with
//information about the imposed displacement and the sizes of the compressed
//coordinate and energy data for the vertices of the network-mesh data
//structure.
struct RelaxationStep {

    RelaxationStep(){}

    double dx, dy, dz;
    //TODO: I am considering adding records of energies, but I suspect it
    //may be better to compute those as needed given the knowledge of
    //the positions and mechanical parameters.
    int xsize;
};

struct NMCRelaxationRun {

    NMCRelaxationRun(){}

    RelaxParams params;
    vector<RelaxationStep> steps;
    string nmc_name;
};

//Tools for reading and writing the intial state of a network

//The following two functions are appropriate when the vertices of the network
//are a subset of the vertices of the simplicial mesh.
bool extract_network_mesh_complex(char *file_name, vector<double> &coordinates, vector<int> &edges, vector<int> &triples, vector<double> &angles, vector<int> &tets);

bool report_network_mesh_complex(char *file_name, shared_ptr<NetworkMeshComplex> nmc);

//The following two functions are appropriate when the vertices of the network
//and the vertices of the simplicial meshes are disjoint sets
bool report_disjoint_network_mesh_complex(char *file_name, shared_ptr<RandomNMCBase> rnmc);

bool extract_disjoint_network_mesh_complex(char *file_name, vector<double> &coordinates, unsigned &num_net_verts, vector<int> &edges, vector<int> &triples, vector<double> &angles, vector<int> &tets, vector<int> &vert_ranges, vector<int> &contained_verts);

//Open a manifest describing a relaxation run
bool open_network_mesh_trajectory(string output, string nmc_file, RelaxParams params, string &xml_name, string &nmt_name);

//Append a state of the network-mesh complex
bool append_network_mesh_state(Vec3D disp, vector<double> coordinates, string xml_name, string nmt_name);

//Close a manifest describing a relaxation run
bool close_network_mesh_trajectory(string xml_name);

//Import data about a relaxation run. Obtain physical parameters, descriptions
//of each relaxation step, and a handle to the file containing compressed,
//binary records of final positions and forces.
bool import_network_mesh_trajectory(string base_name, NMCRelaxationRun &run, FILE **nmt_handle);

//Utility functions for generating output names
int get_dec_digits(int value);

string report_name(string base, int num_digits, int count, string ext);

//Inflate compressed data
int inf(FILE *source, unsigned size, void *dest);
