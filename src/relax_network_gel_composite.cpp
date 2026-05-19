/*
 * This code reads in a Network Mesh Composite (.nmc) file, which contains
 * a description of a discrete network and a tetrahedral finite element mesh,
 * along with a set of material properties, applies a specified program of
 * compression and shear deformations, and relaxes the system to a minimum
 * strain energy state.
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <map>
#include <set>
#include "geometric_primitives.hpp"
#include "file_lists.hpp"
#include <stdlib.h>
#include <iomanip>
#include <cfloat>
#include <unistd.h>
#include <mkl_cblas.h>
#include <time.h>
#include "network_mesh_io.hpp"
#include <unordered_set>
#include <functional>
#include <numeric>
#include <tbb.h>

//Default number of maximum steps for Newton-Raphson root finding
#define DEFAULT_MAX_STEPS 1000

//Size of a trial step when estimating the curvature of the energy along
//a search direction
#define FINITE_DIFF_COEFF 1e-3

#define min(a,b) a <= b ? a : b
#define max(a,b) a >= b ? a : b

//The small dilation to apply to the network to obtain its bulk modulus
#define TRIAL_D 1e-6

#define BENDING_TOL 1e-12

using namespace std;

//Struct to describe a set of displacementss
struct DisplacementProgram {

    //Coarse displacement and optional fine increment
    Vec3D coarse_step, fine_step;
    //Number of coarse_grained fine-grained steps in either direction
    int num_coarse, num_fine;

    DisplacementProgram(Vec3D my_cs, int my_nc){
        coarse_step = my_cs;
        num_coarse = my_nc;
	    num_fine = 0;
    }

    DisplacementProgram(Vec3D my_coarse, int my_nc, Vec3D my_fine, int my_nf){
        coarse_step = my_coarse;
        num_coarse = my_nc;
        fine_step = my_fine;
	    num_fine = my_nf;
    }
};

//Struct giving parameters for nonlinear conjugate gradient optimization
struct NLCGParams {

    NLCGParams(){}

    NLCGParams(int max_cgi, int max_lsi, double my_tol){
        max_cg_iter = max_cgi;
	    max_ls_iter = max_lsi;
        tol = my_tol;
    }

    int max_cg_iter, max_ls_iter;
    double tol;
};

//Struct for holding parameters for the Steihaug Conjugate Gradient-Trust Region
//minimization scheme
struct CGTRParams {

    CGTRParams(){}

    //Maximum numbers of outer iterations and trust region searches
    int max_outer, max_inner;
    //Maximum trust radius, minimum improvement in the cost function relative
    //to the model prediction, and convergence criterion for relative reduction
    //in the residual.
    double max_radius, min_ratio, conv_tol;
};

//Utility function for the parallel summation of arrays
template <typename T> T sum_reduce(const T* first, const T* last, T identity){
    return oneapi::tbb::parallel_reduce(
        // Index range for reduction
        oneapi::tbb::blocked_range<const T*>(first,last),
        // Identity element
        identity,
        // Reduce a subrange and partial sum
        [&]( oneapi::tbb::blocked_range<const T*> r, T partial_sum )->double {
            return std::accumulate( r.begin(), r.end(), partial_sum );
        },
        // Reduce two partial sums
        std::plus<T>()
    );
}

//Utility function for partitioning jobs accross threads
vector<int> job_assignments(int job_num, int thread_num){

    int job_size, remainder, start, iter;
    std::vector<int> assignments;

    job_size = job_num / thread_num;
    remainder = job_num - job_size * thread_num;

    start = 0;
    for(iter = 0; iter < thread_num; iter++){
        assignments.push_back(start);
        start += iter < remainder ? job_size + 1 : job_size;
    }

    assignments.push_back(job_num);

    return assignments;
}

//Load coordinates into a vector
void load_coords(vector<double> points, int idx, Vec3D &pt){

    pt.x = points[3*idx];
    pt.y = points[3*idx+1];
    pt.z = points[3*idx+2];
}

//Compute the volumes of tetrahedra
void calc_volumes(vector<double> points, vector<int> tets, vector<double> &volumes){

    Vec3D p1, p2, p3, p4, d1, d2, d3, cross_prod;

    for(int iter = 0; iter < tets.size() / 4; iter ++){
        load_coords(points, tets[iter*4], p1);
        load_coords(points, tets[iter*4+1], p2);
        load_coords(points, tets[iter*4+2], p3);
        load_coords(points, tets[iter*4+3], p4);

        difference(p2, p1, d1);
        difference(p3, p1, d2);
        difference(p4, p1, d3);

        cross(d2, d3, cross_prod);

        volumes.push_back(abs(dot(d1, cross_prod)) / 6);
    }
}

//Given a set of tetrahedra, with vertex coordinates p1, p2, p3 and p4, find the
//first three columns of the inverse of the matrix
//x1 x2 x3 x4
//y1 y2 y3 y4
//z1 z2 z3 z4
// 1  1  1  1
//
//and store these columns for use in computing the affine transformation matrix
//describing the transformation from the reference to the deformed state.
void get_inv_mats(vector<double> points, vector<int> tets, vector<double> &inv_mats){

    int iter;
    double x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, inv_det;
    double f11, f12, f13, f21, f22, f23, f31, f32, f33;
    int idx1, idx2, idx3, idx4;
    vector<int> inv_mat;

    for(iter = 0; iter < tets.size() / 4; iter ++){
        x1 = points[3*tets[iter*4]];
        y1 = points[3*tets[iter*4] + 1];
        z1 = points[3*tets[iter*4] + 2];

        x2 = points[3*tets[iter*4 + 1]];
        y2 = points[3*tets[iter*4 + 1] + 1];
        z2 = points[3*tets[iter*4 + 1] + 2];

        x3 = points[3*tets[iter*4 + 2]];
        y3 = points[3*tets[iter*4 + 2] + 1];
        z3 = points[3*tets[iter*4 + 2] + 2];

        x4 = points[3*tets[iter*4 + 3]];
        y4 = points[3*tets[iter*4 + 3] + 1];
        z4 = points[3*tets[iter*4 + 3] + 2];

	    inv_det = x1*(y2*(z3 - z4) + y3*(z4 - z2) + y4*(z2 - z3));
	    inv_det += x2*(y1*(z4 - z3) + y3*(z1 - z4) + y4*(z3 - z1));
	    inv_det += x3*(y1*(z2 - z4) + y2*(z4 - z1) + y4*(z1 - z2));
	    inv_det += x4*(y1*(z3 - z2) + y2*(z1 - z3) + y3*(z2 - z1));
	    inv_det = 1 / inv_det;

        inv_mats.push_back((y2*(z3 - z4) + y3*(z4 - z2) + y4*(z2 - z3)) * inv_det);
        inv_mats.push_back((x2*(z4 - z3) + x3*(z2 - z4) + x4*(z3 - z2)) * inv_det);
        inv_mats.push_back((x4*(y2 - y3) + x2*(y3 - y4) + x3*(-y2 + y4)) * inv_det);
        inv_mats.push_back((y1*(z4 - z3) + y3*(z1 - z4) + y4*(z3 - z1)) * inv_det);
        inv_mats.push_back((x1*(z3 - z4) + x3*(z4 - z1) + x4*(z1 - z3)) * inv_det);
        inv_mats.push_back((x1*(y4 - y3) + x3*(y1 - y4) + x4*(y3 - y1)) * inv_det);
        inv_mats.push_back((y1*(z2 - z4) + y2*(z4 - z1) + y4*(z1 - z2)) * inv_det);
        inv_mats.push_back((x1*(z4 - z2) + x2*(z1 - z4) + x4*(z2 - z1)) * inv_det);
        inv_mats.push_back((x1*(y2 - y4) + x2*(y4 - y1) + x4*(y1 - y2)) * inv_det);
        inv_mats.push_back((y1*(z3 - z2) + y2*(z1 - z3) + y3*(z2 - z1)) * inv_det);
        inv_mats.push_back((x1*(z2 - z3) + x2*(z3 - z1) + x3*(z1 - z2)) * inv_det);
        inv_mats.push_back((x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1)) * inv_det);
    }
}

//Parse an NMC file. Obtain vertex bounds, vertices, edges, resting bond 
//lengths, triples, mesh elements, element volumes, and matrices to be used to 
//compute deformation gradient tensors.
//bool parse_nmc_file(string file_name, vector<double> &bounds, vector<double> &points, unordered_set<int> &bottom, unordered_set<int> &top, vector<int> &edges, vector<double> &lengths, vector<int> &triples, vector<double> &angles, vector<int> &tetrahedra, vector<double> &volumes, vector<double> &inv_mats){
bool parse_nmc_file(string file_name, vector<double> &bounds, vector<double> &points, unordered_set<int> &bottom, unordered_set<int> &top, vector<int> &edges, vector<double> &lengths, vector<int> &triples, vector<double> &angles, vector<int> &tetrahedra){

    double minx = FLT_MAX, maxx = -FLT_MAX, miny = FLT_MAX, maxy = -FLT_MAX;
    double minz = FLT_MAX, maxz = -FLT_MAX;
    int iter, idx1, idx2;

    //Read compressed, binary data from the nmc file
    if(! extract_network_mesh_complex((char *) file_name.c_str(), points, edges, triples, angles, tetrahedra)){
        cerr << "The network-mesh extraction failed.\n";
	    return false;
    }

    //Find the extremes along each direction and identify top and bottom points
    for(iter = 0; iter < points.size(); iter += 3){
        if(points[iter] < minx) minx = points[iter];
        if(points[iter+1] < miny) miny = points[iter+1];
        if(points[iter+2] < minz) minz = points[iter+2];
        if(points[iter] > maxx) maxx = points[iter];
        if(points[iter+1] > maxy) maxy = points[iter+1];
        if(points[iter+2] > maxz) maxz = points[iter+2];
    }

    bounds.push_back(minx);
    bounds.push_back(maxx);
    bounds.push_back(miny);
    bounds.push_back(maxy);
    bounds.push_back(minz);
    bounds.push_back(maxz);

    for(iter = 2; iter < points.size(); iter += 3){
        if(abs(points[iter] - minz) < TOL) bottom.insert(iter / 3);
        if(abs(points[iter] - maxz) < TOL) top.insert(iter / 3);
    }

    //Find resting edge lengths
    for(iter = 0; iter < edges.size(); iter += 2){
        idx1 = edges[iter];
        idx2 = edges[iter+1];
        Vec3D p1(points[3*idx1], points[3*idx1+1], points[3*idx1+2]);
        Vec3D p2(points[3*idx2], points[3*idx2+1], points[3*idx2+2]);
	    lengths.push_back(distance(p1, p2));
    }

    //Find tetrahedron volumes
    //calc_volumes(points, tetrahedra, volumes);

    //Find matrices for computing deformation gradient tensors
    //get_inv_mats(points, tetrahedra, inv_mats);

    return true;
}

//Compute fiber stretching forces in a serial manner.
void stretching_forces(vector<double> points, vector<int> edges, vector<double> lengths, double k, vector<double> &forces, int start, int end){

    Vec3D r1, r2, force;
    double dist;
    int iter, idx1, idx2;

    for(iter = start; iter != end; iter ++){

        idx1 = edges[2*iter];
        idx2 = edges[2*iter + 1];

        r1.x = points[3*idx1];
        r1.y = points[3*idx1 + 1];
        r1.z = points[3*idx1 + 2];

        r2.x = points[3*idx2];
        r2.y = points[3*idx2 + 1];
        r2.z = points[3*idx2 + 2];

        difference(r2, r1, force);
        dist = distance(r1, r2);
        scalar_multiply(force, k*(1 - lengths[iter]/dist));

        forces[3*idx1] += force.x;
        forces[3*idx1 + 1] += force.y;
        forces[3*idx1 + 2] += force.z;
        forces[3*idx2] -= force.x;
        forces[3*idx2 + 1] -= force.y;
        forces[3*idx2 + 2] -= force.z;
    }
}

//Compute fiber stretching forces in a parallel manner.
/*void stretching_forces_parallel(vector<double> points, vector<int> edges, vector<double> lengths, double k, int edge_idx, vector<tbb::concurrent_vector<double>> &force_records){

    Vec3D r1, r2, force;
    double dist;
    int iter, idx1, idx2;

    idx1 = edges[2*edge_idx]; 
    idx2 = edges[2*edge_idx + 1];

    r1.x = points[3*idx1]; 
    r1.y = points[3*idx1 + 1];
    r1.z = points[3*idx1 + 2];

    r2.x = points[3*idx2]; 
    r2.y = points[3*idx2 + 1];
    r2.z = points[3*idx2 + 2];

    difference(r2, r1, force);
    dist = distance(r1, r2);
    scalar_multiply(force, k*(1 - lengths[edge_idx]/dist));

    force_records[3*idx1].push_back(force.x);
    force_records[3*idx1+1].push_back(force.y);
    force_records[3*idx1+2].push_back(force.z);
    scalar_multiply(force, -1);
    force_records[3*idx2].push_back(force.x);
    force_records[3*idx2+1].push_back(force.y);
    force_records[3*idx2+2].push_back(force.z);
}*/

//Compute fiber stretching energy in a serial manner.
double stretching_energy(vector<double> points, vector<int> edges, vector<double> lengths, double k){

    Vec3D r1, r2;
    double dist, energy = 0;
    int iter, idx1, idx2;

    for(iter = 0; iter < edges.size() / 2; iter ++){

        idx1 = edges[2*iter];
        idx2 = edges[2*iter + 1];

        r1.x = points[3*idx1];
        r1.y = points[3*idx1 + 1];
        r1.z = points[3*idx1 + 2];

        r2.x = points[3*idx2];
        r2.y = points[3*idx2 + 1];
        r2.z = points[3*idx2 + 2];

        dist = distance(r1, r2);
        energy += .5 * k * (dist - lengths[iter])*(dist - lengths[iter]);
    }

    return energy;
}

//Helper function for computing fiber stretching energy in a parallel manner.
void stretching_energy_worker(vector<double> points, vector<int> edges, vector<double> lengths, double k, int index, double *energy){

    Vec3D r1, r2;
    double dist;
    int idx1, idx2;

    idx1 = edges[2*index];
    idx2 = edges[2*index + 1];

    r1.x = points[3*idx1];
    r1.y = points[3*idx1 + 1];
    r1.z = points[3*idx1 + 2];

    r2.x = points[3*idx2];
    r2.y = points[3*idx2 + 1];
    r2.z = points[3*idx2 + 2];

    dist = distance(r1, r2);
    *energy = .5 * k * (dist - lengths[index])*(dist - lengths[index]);
}

double parallel_stretching_energy(vector<double> points, vector<int> edges, vector<double> lengths, double k){

    vector<double> energies(edges.size() / 2, 0);

    //Compute energies in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, edges.size()/2),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                stretching_energy_worker(points, edges, lengths, k, index, &energies[index]);
            }
        }
    );

    //Use a parallel reduction to find the total energy
    return sum_reduce(&energies[0], &energies[0]+energies.size(), 0.);
}

//Compute fiber bending forces in a serial manner.
void bending_forces(vector<double> pts, vector<int> triples, vector<double> angles, double kb, vector<double> &forces, int start, int end) {

    Vec3D r1, r2, r3, e1, e2, e3, r_21, r_23, f1, f2, f3;
    double ang, theta, phi, e3mag;
    int iter, idx1, idx2, idx3;

    for(iter = start; iter != end; iter ++){

        idx1 = triples[3*iter];
        idx2 = triples[3*iter+1];
	    idx3 = triples[3*iter+2];

        //Load relevant points
        r1.x = pts[3*idx1];
        r1.y = pts[3*idx1 + 1];
        r1.z = pts[3*idx1 + 2];

        r2.x = pts[3*idx2];
        r2.y = pts[3*idx2 + 1];
        r2.z = pts[3*idx2 + 2];

        r3.x = pts[3*idx3];
        r3.y = pts[3*idx3 + 1];
        r3.z = pts[3*idx3 + 2];

        //Find normal std::vectors from the hinge to each end point
        difference(r1, r2, r_21);
        normalize(r_21);
        difference(r3, r2, r_23);
        normalize(r_23);

        //Compute an orthogonal triad
        difference(r1, r2, e1);
        normalize(e1);

        cross(e1, r_23, e3);

        //If the edges are parallel or very nearly so, either the fiber is
        //by default straight and is unbent (the likelier scenario), or the
        //fiber is not perfectly straight by default, in which case an arbitrary
        //plane normal vector must be computed.
        e3mag = magnitude(e3);
        if(e3mag < BENDING_TOL){
            if(abs(angles[iter] - M_PI) < BENDING_TOL) continue;

            theta = acos(e3.z / e3mag);
            theta = theta < M_PI / 2 ? theta + M_PI / 2 : theta - M_PI / 2;
            phi = atan2(e3.y, e3.x);
            e3.x = sin(theta)*cos(phi);
            e3.y = sin(theta)*sin(phi);
            e3.z = cos(theta);
        } 

        else normalize(e3);

        cross(e3, e1, e2);
        normalize(e2);

        //Find a bending angle
        ang = atan2(dot(e2, r_23), dot(e1, r_23));
        if(ang < 0) ang += 2*M_PI;

        //Compute the force on the first index
        cross(e3, r_21, f1);
        scalar_multiply(f1, kb*(ang-angles[iter])/distance(r1, r2));

        //Compute the force on the last index
        cross(r_23, e3, f3);
        scalar_multiply(f3, kb*(ang-angles[iter])/distance(r3, r2));

        //Compute the force on the hinge index by computing minus the sum
        //of the other two.
        sum(f1, f3, f2);
        scalar_multiply(f2, -1);

        forces[idx1*3] += f1.x;
        forces[idx1*3+1] += f1.y;
        forces[idx1*3+2] += f1.z;
        forces[idx2*3] += f2.x;
        forces[idx2*3+1] += f2.y;
        forces[idx2*3+2] += f2.z;
        forces[idx3*3] += f3.x;
        forces[idx3*3+1] += f3.y;
        forces[idx3*3+2] += f3.z;
    }
}

//Compute fiber bending forces in a parallel manner.
/*void bending_forces_parallel(vector<double> pts, vector<int> triples, vector<double> angles, double kb, int trip_idx, vector<tbb::concurrent_vector<double>> &f_records) {

    Vec3D r1, r2, r3, e1, e2, e3, r_21, r_23, f1, f2, f3;
    double ang, e3mag, theta, phi;
    int idx1, idx2, idx3;

    idx1 = triples[3*trip_idx];
    idx2 = triples[3*trip_idx+1];
    idx3 = triples[3*trip_idx+2];

    //Load relevant points
    r1.x = pts[3*idx1];
    r1.y = pts[3*idx1 + 1];
    r1.z = pts[3*idx1 + 2];

    r2.x = pts[3*idx2];
    r2.y = pts[3*idx2 + 1];
    r2.z = pts[3*idx2 + 2];

    r3.x = pts[3*idx3];
    r3.y = pts[3*idx3 + 1];
    r3.z = pts[3*idx3 + 2];

    //Find normal std::vectors from the hinge to each end point
    difference(r1, r2, r_21);
    normalize(r_21);
    difference(r3, r2, r_23);
    normalize(r_23);

    //Compute an orthogonal triad
    difference(r1, r2, e1);
    normalize(e1);

    cross(e1, r_23, e3);

    //If the edges are parallel or very nearly so, either the fiber is
    //by default straight and is unbent (the likelier scenario), or the
    //fiber is not perfectly straight by default, in which case an arbitrary
    //plane normal vector must be computed.
    e3mag = magnitude(e3);
    if(e3mag < BENDING_TOL){
        if(abs(angles[trip_idx] - M_PI) < BENDING_TOL){
            f_records[3*idx1].push_back(0);
            f_records[3*idx1+1].push_back(0);
            f_records[3*idx1+2].push_back(0);
            f_records[3*idx2].push_back(0);
            f_records[3*idx2+1].push_back(0);
            f_records[3*idx2+2].push_back(0);
            f_records[3*idx3].push_back(0);
            f_records[3*idx3+1].push_back(0);
            f_records[3*idx3+2].push_back(0);
            return;
        }

        theta = acos(e3.z / e3mag);
        theta = theta < M_PI / 2 ? theta + M_PI / 2 : theta - M_PI / 2;
        phi = atan2(e3.y, e3.x);
        e3.x = sin(theta)*cos(phi);
        e3.y = sin(theta)*sin(phi);
        e3.z = cos(theta);
    } 

    else normalize(e3);

    cross(e3, e1, e2);
    normalize(e2);

    //Find a bending angle
    ang = atan2(dot(e2, r_23), dot(e1, r_23));
    if(ang < 0) ang += 2*M_PI;

    //Compute the force on the first index
    cross(e3, r_21, f1);
    scalar_multiply(f1, kb*(ang-angles[trip_idx])/distance(r1, r2));

    //Compute the force on the last index
    cross(r_23, e3, f3);
    scalar_multiply(f3, kb*(ang-angles[trip_idx])/distance(r3, r2));

    //Compute the force on the hinge index by computing minus the sum
    //of the other two.
    sum(f1, f3, f2);
    scalar_multiply(f2, -1);

    f_records[3*idx1].push_back(f1.x);
    f_records[3*idx1+1].push_back(f1.y);
    f_records[3*idx1+2].push_back(f1.z);

    f_records[3*idx2].push_back(f2.x);
    f_records[3*idx2+1].push_back(f2.y);
    f_records[3*idx2+2].push_back(f2.z);

    f_records[3*idx3].push_back(f3.x);
    f_records[3*idx3+1].push_back(f3.y);
    f_records[3*idx3+2].push_back(f3.z);
}*/

//Compute fiber bending energy in a serial manner.
double fiber_bending_energy(vector<double> points, vector<int> triples, vector<double> angles, double kb){

    Vec3D r1, r2, r3, e1, e2, e3, r_23;
    double e3mag, ang, theta, phi, energy = 0;
    int iter, idx1, idx2, idx3;

    for(iter = 0; iter < triples.size() / 3; iter ++){

        idx1 = triples[3*iter];
        idx2 = triples[3*iter+1];
	    idx3 = triples[3*iter+2];

        //Load relevant points
        r1.x = points[3*idx1];
        r1.y = points[3*idx1 + 1];
        r1.z = points[3*idx1 + 2];

        r2.x = points[3*idx2];
        r2.y = points[3*idx2 + 1];
        r2.z = points[3*idx2 + 2];

        r3.x = points[3*idx3];
        r3.y = points[3*idx3 + 1];
        r3.z = points[3*idx3 + 2];

        //Find normal vectors from the hinge to each end point
        difference(r1, r2, e1);
        normalize(e1);
        difference(r3, r2, r_23);
        normalize(r_23);

        //Compute an orthogonal triad

        cross(e1, r_23, e3);

        //If the edges are parallel or very nearly so, either the fiber is
        //by default straight and is unbent (the likelier scenario), or the
        //fiber is not perfectly straight by default, in which case an arbitrary
        //plane normal vector must be computed.
        e3mag = magnitude(e3);
        if(e3mag < BENDING_TOL){
            if(abs(angles[iter] - M_PI) < BENDING_TOL) continue;

            theta = acos(e3.z / e3mag);
            theta = theta < M_PI / 2 ? theta + M_PI / 2 : theta - M_PI / 2;
            phi = atan2(e3.y, e3.x);
            e3.x = sin(theta)*cos(phi);
            e3.y = sin(theta)*sin(phi);
            e3.z = cos(theta);
        } 

        else normalize(e3);

        cross(e3, e1, e2);
        normalize(e2);

        //Find a bending angle
        ang = atan2(dot(e2, r_23), dot(e1, r_23));
        if(ang < 0) ang += 2*M_PI;
        energy += .5*kb*(ang-angles[iter])*(ang-angles[iter]);
    }

    return energy;
}

//Compute fiber bending energies in a parallel manner.
void fiber_bending_energy_worker(vector<double> points, vector<int> triples, vector<double> angles, double kb, int index, double *energy){

    Vec3D r1, r2, r3, e1, e2, e3, r_23;
    double ang, theta, phi, e3mag;
    int idx1, idx2, idx3;

    idx1 = triples[3*index];
    idx2 = triples[3*index+1];
    idx3 = triples[3*index+2];

    //Load relevant points
    r1.x = points[3*idx1];
    r1.y = points[3*idx1 + 1];
    r1.z = points[3*idx1 + 2];

    r2.x = points[3*idx2];
    r2.y = points[3*idx2 + 1];
    r2.z = points[3*idx2 + 2];

    r3.x = points[3*idx3];
    r3.y = points[3*idx3 + 1];
    r3.z = points[3*idx3 + 2];

    //Find normal vectors from the hinge to each end point
    difference(r1, r2, e1);
    normalize(e1);
    difference(r3, r2, r_23);
    normalize(r_23);

    //Compute an orthogonal triad
    cross(e1, r_23, e3);

    //If the edges are parallel or very nearly so, either the fiber is
    //by default straight and is unbent (the likelier scenario), or the
    //fiber is not perfectly straight by default, in which case an arbitrary
    //plane normal vector must be computed.
    e3mag = magnitude(e3);
    if(e3mag < BENDING_TOL){
        if(abs(angles[index] - M_PI) < BENDING_TOL){
            *energy = 0;
            return;
        }

        theta = acos(e3.z / e3mag);
        theta = theta < M_PI / 2 ? theta + M_PI / 2 : theta - M_PI / 2;
        phi = atan2(e3.y, e3.x);
        e3.x = sin(theta)*cos(phi);
        e3.y = sin(theta)*sin(phi);
        e3.z = cos(theta);
    } 

    else normalize(e3);

    cross(e3, e1, e2);
    normalize(e2);

    //Find a bending angle
    ang = atan2(dot(e2, r_23), dot(e1, r_23));
    if(ang < 0) ang += 2*M_PI;
    *energy = .5*kb*(ang-angles[index])*(ang-angles[index]);
}

double parallel_fiber_bending_energy(vector<double> points, vector<int> triples,vector<double> angles, double kb){

    vector<double> energies(triples.size() / 3, 0);

    //Compute energies in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, triples.size()/3),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                fiber_bending_energy_worker(points, triples, angles, kb, index, &energies[index]);
            }
        }
    );

    //Sum energies using a parallel reduction
    return sum_reduce(&energies[0], &energies[0]+energies.size(), 0.);
}

//Find gel restoring forces in a serial manner.
void gel_forces(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi, vector<double> &forces, int start, int end){

    //Components of the deformation gradient tensor, and derivatives of the
    //gel elastic energy with respect to the deformation gradient tensor
    //components.
    double f11, f12, f13, f21, f22, f23, f31, f32, f33, de_df11, de_df12;
    double de_df13, de_df21, de_df22, de_df23, de_df31, de_df32, de_df33;

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and coefficients multiplying deformation gradient tensor
    //components to calculate forces
    double det, sq_sum, coeff1, coeff2;
    int idx1, idx2, idx3, idx4;

    vector<double> inv_mat;

    for(int tet_idx = start; tet_idx != end; tet_idx ++){

        //Integer indices of the vertices of the tetrahedron
        idx1 = tetrahedra[tet_idx*4];
        idx2 = tetrahedra[tet_idx*4+1];
        idx3 = tetrahedra[tet_idx*4+2];
        idx4 = tetrahedra[tet_idx*4+3];

        //Matrix for computing an affine transformation for the current 
	    //tetrahedral cell
	    inv_mat.assign(inv_mats.begin()+12*tet_idx, inv_mats.begin()+12*(tet_idx+1));

        //Find the components of the deformation gradient tensor
        f11 = pts[idx1*3]*inv_mat[0] + pts[idx2*3]*inv_mat[3] + pts[idx3*3]*inv_mat[6] + pts[idx4*3]*inv_mat[9];
        f12 = pts[idx1*3]*inv_mat[1] + pts[idx2*3]*inv_mat[4] + pts[idx3*3]*inv_mat[7] + pts[idx4*3]*inv_mat[10];
        f13 = pts[idx1*3]*inv_mat[2] + pts[idx2*3]*inv_mat[5] + pts[idx3*3]*inv_mat[8] + pts[idx4*3]*inv_mat[11];

        f21 = pts[idx1*3+1]*inv_mat[0] + pts[idx2*3+1]*inv_mat[3] + pts[idx3*3+1]*inv_mat[6] + pts[idx4*3+1]*inv_mat[9];
        f22 = pts[idx1*3+1]*inv_mat[1] + pts[idx2*3+1]*inv_mat[4] + pts[idx3*3+1]*inv_mat[7] + pts[idx4*3+1]*inv_mat[10];
        f23 = pts[idx1*3+1]*inv_mat[2] + pts[idx2*3+1]*inv_mat[5] + pts[idx3*3+1]*inv_mat[8] + pts[idx4*3+1]*inv_mat[11];

        f31 = pts[idx1*3+2]*inv_mat[0] + pts[idx2*3+2]*inv_mat[3] + pts[idx3*3+2]*inv_mat[6] + pts[idx4*3+2]*inv_mat[9];
        f32 = pts[idx1*3+2]*inv_mat[1] + pts[idx2*3+2]*inv_mat[4] + pts[idx3*3+2]*inv_mat[7] + pts[idx4*3+2]*inv_mat[10];
        f33 = pts[idx1*3+2]*inv_mat[2] + pts[idx2*3+2]*inv_mat[5] + pts[idx3*3+2]*inv_mat[8] + pts[idx4*3+2]*inv_mat[11];

        //Find derivatives of the gel elastic energy with respect to the 
	    //components of the deformation gradient tensor.
        det = f11*(f22*f33-f23*f23) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);
        sq_sum = f11*f11 + f12*f12 + f13*f13 + f21*f21 + f22*f22 + f23*f23;
        sq_sum += f31*f31 + f32*f32 + f33*f33;
        coeff1 = -g*volumes[tet_idx];
        coeff2 = volumes[tet_idx] * (g*(1 + excess_osm) / det - kt_o_omega * (log((det - 1) / det) + (det+chi)/(det*det)));
        /*if(coeff2 != coeff2){
            cout << "NaN found at index " << tet_idx << endl;
            cout << "Determinant: " << det << endl;
        }*/

        de_df11 = coeff1*f11 + coeff2*(f22*f33 - f23*f32);
        de_df12 = coeff1*f12 + coeff2*(f23*f31 - f21*f33);
        de_df13 = coeff1*f13 + coeff2*(f21*f32 - f22*f31);
        de_df21 = coeff1*f21 + coeff2*(f13*f32 - f12*f33);
        de_df22 = coeff1*f22 + coeff2*(f11*f33 - f13*f31);
        de_df23 = coeff1*f23 + coeff2*(f12*f31 - f11*f32);
        de_df31 = coeff1*f31 + coeff2*(f12*f23 - f13*f22);
        de_df32 = coeff1*f32 + coeff2*(f13*f21 - f11*f23);
        de_df33 = coeff1*f33 + coeff2*(f11*f22 - f12*f21);

        //Find forces on each vertex using the chain rule
        forces[idx1*3] += de_df11*inv_mat[0] + de_df12*inv_mat[1] + de_df13*inv_mat[2];
        forces[idx1*3+1] += de_df21*inv_mat[0] + de_df22*inv_mat[1] + de_df23*inv_mat[2];
        forces[idx1*3+2] += de_df31*inv_mat[0] + de_df32*inv_mat[1] + de_df33*inv_mat[2];

        forces[idx2*3] += de_df11*inv_mat[3] + de_df12*inv_mat[4] + de_df13*inv_mat[5];
        forces[idx2*3+1] += de_df21*inv_mat[3] + de_df22*inv_mat[4] + de_df23*inv_mat[5];
        forces[idx2*3+2] += de_df31*inv_mat[3] + de_df32*inv_mat[4] + de_df33*inv_mat[5];

        forces[idx3*3] += de_df11*inv_mat[6] + de_df12*inv_mat[7] + de_df13*inv_mat[8];
        forces[idx3*3+1] += de_df21*inv_mat[6] + de_df22*inv_mat[7] + de_df23*inv_mat[8];
        forces[idx3*3+2] += de_df31*inv_mat[6] + de_df32*inv_mat[7] + de_df33*inv_mat[8];

        forces[idx4*3] += de_df11*inv_mat[9] + de_df12*inv_mat[10] + de_df13*inv_mat[11];
        forces[idx4*3+1] += de_df21*inv_mat[9] + de_df22*inv_mat[10] + de_df23*inv_mat[11];
        forces[idx4*3+2] += de_df31*inv_mat[9] + de_df32*inv_mat[10] + de_df33*inv_mat[11];
    }
}

//Find gel restoring forces in a parallel manner.
/*void gel_forces_parallel(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi, int tet_idx, vector<tbb::concurrent_vector<double>> &f_records){

    //Components of the deformation gradient tensor, and derivatives of the
    //gel elastic energy with respect to the deformation gradient tensor
    //components.
    double f11, f12, f13, f21, f22, f23, f31, f32, f33, de_df11, de_df12;
    double de_df13, de_df21, de_df22, de_df23, de_df31, de_df32, de_df33;

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and coefficients multiplying deformation gradient tensor
    //components to calculate forces
    double det, sq_sum, coeff1, coeff2, fx, fy, fz;
    int idx1, idx2, idx3, idx4;

    vector<double> inv_mat;

    //Integer indices of the vertices of the tetrahedron
    idx1 = tetrahedra[tet_idx*4];
    idx2 = tetrahedra[tet_idx*4+1];
    idx3 = tetrahedra[tet_idx*4+2];
    idx4 = tetrahedra[tet_idx*4+3];

    //Matrix for computing an affine transformation for the current 
    //tetrahedral cell
    inv_mat.assign(inv_mats.begin()+12*tet_idx,inv_mats.begin()+12*(tet_idx+1));

    //Find the components of the deformation gradient tensor
    f11 = pts[idx1*3]*inv_mat[0] + pts[idx2*3]*inv_mat[3] + pts[idx3*3]*inv_mat[6] + pts[idx4*3]*inv_mat[9];
    f12 = pts[idx1*3]*inv_mat[1] + pts[idx2*3]*inv_mat[4] + pts[idx3*3]*inv_mat[7] + pts[idx4*3]*inv_mat[10];
    f13 = pts[idx1*3]*inv_mat[2] + pts[idx2*3]*inv_mat[5] + pts[idx3*3]*inv_mat[8] + pts[idx4*3]*inv_mat[11];

    f21 = pts[idx1*3+1]*inv_mat[0] + pts[idx2*3+1]*inv_mat[3] + pts[idx3*3+1]*inv_mat[6] + pts[idx4*3+1]*inv_mat[9];
    f22 = pts[idx1*3+1]*inv_mat[1] + pts[idx2*3+1]*inv_mat[4] + pts[idx3*3+1]*inv_mat[7] + pts[idx4*3+1]*inv_mat[10];
    f23 = pts[idx1*3+1]*inv_mat[2] + pts[idx2*3+1]*inv_mat[5] + pts[idx3*3+1]*inv_mat[8] + pts[idx4*3+1]*inv_mat[11];

    f31 = pts[idx1*3+2]*inv_mat[0] + pts[idx2*3+2]*inv_mat[3] + pts[idx3*3+2]*inv_mat[6] + pts[idx4*3+2]*inv_mat[9];
    f32 = pts[idx1*3+2]*inv_mat[1] + pts[idx2*3+2]*inv_mat[4] + pts[idx3*3+2]*inv_mat[7] + pts[idx4*3+2]*inv_mat[10];
    f33 = pts[idx1*3+2]*inv_mat[2] + pts[idx2*3+2]*inv_mat[5] + pts[idx3*3+2]*inv_mat[8] + pts[idx4*3+2]*inv_mat[11];

    //Find derivatives of the gel elastic energy with respect to the 
    //components of the deformation gradient tensor.
    det = f11*(f22*f33-f23*f23) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);
    sq_sum = f11*f11 + f12*f12 + f13*f13 + f21*f21 + f22*f22 + f23*f23;
    sq_sum += f31*f31 + f32*f32 + f33*f33;
    coeff1 = -g*volumes[tet_idx];
    coeff2 = volumes[tet_idx] * (g*(1 + excess_osm) / det - kt_o_omega * (log((det - 1) / det) + (det+chi)/(det*det)));

    de_df11 = coeff1*f11 + coeff2*(f22*f33 - f23*f32);
    de_df12 = coeff1*f12 + coeff2*(f23*f31 - f21*f33);
    de_df13 = coeff1*f13 + coeff2*(f21*f32 - f22*f31);
    de_df21 = coeff1*f21 + coeff2*(f13*f32 - f12*f33);
    de_df22 = coeff1*f22 + coeff2*(f11*f33 - f13*f31);
    de_df23 = coeff1*f23 + coeff2*(f12*f31 - f11*f32);
    de_df31 = coeff1*f31 + coeff2*(f12*f23 - f13*f22);
    de_df32 = coeff1*f32 + coeff2*(f13*f21 - f11*f23);
    de_df33 = coeff1*f33 + coeff2*(f11*f22 - f12*f21);

    //Find forces on each vertex using the chain rule
    fx = de_df11*inv_mat[0] + de_df12*inv_mat[1] + de_df13*inv_mat[2];
    fy = de_df21*inv_mat[0] + de_df22*inv_mat[1] + de_df23*inv_mat[2];
    fz = de_df31*inv_mat[0] + de_df32*inv_mat[1] + de_df33*inv_mat[2];
    f_records[3*idx1].push_back(fx);
    f_records[3*idx1+1].push_back(fy);
    f_records[3*idx1+2].push_back(fz);

    fx = de_df11*inv_mat[3] + de_df12*inv_mat[4] + de_df13*inv_mat[5];
    fy = de_df21*inv_mat[3] + de_df22*inv_mat[4] + de_df23*inv_mat[5];
    fz = de_df31*inv_mat[3] + de_df32*inv_mat[4] + de_df33*inv_mat[5];
    f_records[3*idx2].push_back(fx);
    f_records[3*idx2+1].push_back(fy);
    f_records[3*idx2+2].push_back(fz);

    fx = de_df11*inv_mat[6] + de_df12*inv_mat[7] + de_df13*inv_mat[8];
    fy = de_df21*inv_mat[6] + de_df22*inv_mat[7] + de_df23*inv_mat[8];
    fz = de_df31*inv_mat[6] + de_df32*inv_mat[7] + de_df33*inv_mat[8];
    f_records[3*idx3].push_back(fx);
    f_records[3*idx3+1].push_back(fy);
    f_records[3*idx3+2].push_back(fz);

    fx = de_df11*inv_mat[9] + de_df12*inv_mat[10] + de_df13*inv_mat[11];
    fy = de_df21*inv_mat[9] + de_df22*inv_mat[10] + de_df23*inv_mat[11];
    fz = de_df31*inv_mat[9] + de_df32*inv_mat[10] + de_df33*inv_mat[11];
    f_records[3*idx4].push_back(fx);
    f_records[3*idx4+1].push_back(fy);
    f_records[3*idx4+2].push_back(fz);
}*/

//Find the deformation energies of gel elements in a serial manner.
double calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi){

    //Components of the deformation gradient tensor, and derivatives of the
    //gel elastic energy with respect to the deformation gradient tensor
    //components.
    double f11, f12, f13, f21, f22, f23, f31, f32, f33, de_df11, de_df12;
    double de_df13, de_df21, de_df22, de_df23, de_df31, de_df32, de_df33;

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and coefficients multiplying deformation gradient tensor
    //components to calculate forces
    double det, sq_sum, energy = 0;
    int idx1, idx2, idx3, idx4;

    vector<double> inv_mat;

    for(int tet_idx = 0; tet_idx < tetrahedra.size() / 4; tet_idx ++){

        //Integer indices of the vertices of the tetrahedron
        idx1 = tetrahedra[tet_idx*4];
        idx2 = tetrahedra[tet_idx*4+1];
        idx3 = tetrahedra[tet_idx*4+2];
        idx4 = tetrahedra[tet_idx*4+3];

        //Matrix for computing an affine transformation for the current 
	    //tetrahedral cell
	    inv_mat.assign(inv_mats.begin()+12*tet_idx, inv_mats.begin()+12*(tet_idx+1));

        //Find the components of the deformation gradient tensor
        f11 = pts[idx1*3]*inv_mat[0] + pts[idx2*3]*inv_mat[3] + pts[idx3*3]*inv_mat[6] + pts[idx4*3]*inv_mat[9];
        f12 = pts[idx1*3]*inv_mat[1] + pts[idx2*3]*inv_mat[4] + pts[idx3*3]*inv_mat[7] + pts[idx4*3]*inv_mat[10];
        f13 = pts[idx1*3]*inv_mat[2] + pts[idx2*3]*inv_mat[5] + pts[idx3*3]*inv_mat[8] + pts[idx4*3]*inv_mat[11];

        f21 = pts[idx1*3+1]*inv_mat[0] + pts[idx2*3+1]*inv_mat[3] + pts[idx3*3+1]*inv_mat[6] + pts[idx4*3+1]*inv_mat[9];
        f22 = pts[idx1*3+1]*inv_mat[1] + pts[idx2*3+1]*inv_mat[4] + pts[idx3*3+1]*inv_mat[7] + pts[idx4*3+1]*inv_mat[10];
        f23 = pts[idx1*3+1]*inv_mat[2] + pts[idx2*3+1]*inv_mat[5] + pts[idx3*3+1]*inv_mat[8] + pts[idx4*3+1]*inv_mat[11];

        f31 = pts[idx1*3+2]*inv_mat[0] + pts[idx2*3+2]*inv_mat[3] + pts[idx3*3+2]*inv_mat[6] + pts[idx4*3+2]*inv_mat[9];
        f32 = pts[idx1*3+2]*inv_mat[1] + pts[idx2*3+2]*inv_mat[4] + pts[idx3*3+2]*inv_mat[7] + pts[idx4*3+2]*inv_mat[10];
        f33 = pts[idx1*3+2]*inv_mat[2] + pts[idx2*3+2]*inv_mat[5] + pts[idx3*3+2]*inv_mat[8] + pts[idx4*3+2]*inv_mat[11];

        //Find derivatives of the gel elastic energy with respect to the 
	    //components of the deformation gradient tensor.
        det = f11*(f22*f33-f23*f23) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);

        //cout << "Determinant: " << det << endl;

        sq_sum = f11*f11 + f12*f12 + f13*f13 + f21*f21 + f22*f22 + f23*f23;
        sq_sum += f31*f31 + f32*f32 + f33*f33;
        energy += volumes[tet_idx] * g * .5 * (sq_sum - 3 - 2*(1 + excess_osm)*log(det));
        energy += volumes[tet_idx] * kt_o_omega * ((det - 1)*log((det - 1) / det) + chi*(det - 1)/det);
    }

    return energy;
}

//Find the deformation energies of gel elements in a parallel manner.
void gel_energy_worker(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi, int index, double *energy){

    //Components of the deformation gradient tensor, and derivatives of the
    //gel elastic energy with respect to the deformation gradient tensor
    //components.
    double f11, f12, f13, f21, f22, f23, f31, f32, f33, de_df11, de_df12;
    double de_df13, de_df21, de_df22, de_df23, de_df31, de_df32, de_df33;

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and coefficients multiplying deformation gradient tensor
    //components to calculate forces
    double det, sq_sum;
    int idx1, idx2, idx3, idx4;

    vector<double> inv_mat;

    //Integer indices of the vertices of the tetrahedron
    idx1 = tetrahedra[index*4];
    idx2 = tetrahedra[index*4+1];
    idx3 = tetrahedra[index*4+2];
    idx4 = tetrahedra[index*4+3];

    //Matrix for computing an affine transformation for the current 
    //tetrahedral cell
    inv_mat.assign(inv_mats.begin()+12*index, inv_mats.begin()+12*(index+1));

    //Find the components of the deformation gradient tensor
    f11 = pts[idx1*3]*inv_mat[0] + pts[idx2*3]*inv_mat[3] + pts[idx3*3]*inv_mat[6] + pts[idx4*3]*inv_mat[9];
    f12 = pts[idx1*3]*inv_mat[1] + pts[idx2*3]*inv_mat[4] + pts[idx3*3]*inv_mat[7] + pts[idx4*3]*inv_mat[10];
    f13 = pts[idx1*3]*inv_mat[2] + pts[idx2*3]*inv_mat[5] + pts[idx3*3]*inv_mat[8] + pts[idx4*3]*inv_mat[11];

    f21 = pts[idx1*3+1]*inv_mat[0] + pts[idx2*3+1]*inv_mat[3] + pts[idx3*3+1]*inv_mat[6] + pts[idx4*3+1]*inv_mat[9];
    f22 = pts[idx1*3+1]*inv_mat[1] + pts[idx2*3+1]*inv_mat[4] + pts[idx3*3+1]*inv_mat[7] + pts[idx4*3+1]*inv_mat[10];
    f23 = pts[idx1*3+1]*inv_mat[2] + pts[idx2*3+1]*inv_mat[5] + pts[idx3*3+1]*inv_mat[8] + pts[idx4*3+1]*inv_mat[11];

    f31 = pts[idx1*3+2]*inv_mat[0] + pts[idx2*3+2]*inv_mat[3] + pts[idx3*3+2]*inv_mat[6] + pts[idx4*3+2]*inv_mat[9];
    f32 = pts[idx1*3+2]*inv_mat[1] + pts[idx2*3+2]*inv_mat[4] + pts[idx3*3+2]*inv_mat[7] + pts[idx4*3+2]*inv_mat[10];
    f33 = pts[idx1*3+2]*inv_mat[2] + pts[idx2*3+2]*inv_mat[5] + pts[idx3*3+2]*inv_mat[8] + pts[idx4*3+2]*inv_mat[11];

    //Find derivatives of the gel elastic energy with respect to the 
    //components of the deformation gradient tensor.
    det = f11*(f22*f33-f23*f23) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);
    sq_sum = f11*f11 + f12*f12 + f13*f13 + f21*f21 + f22*f22 + f23*f23;
    sq_sum += f31*f31 + f32*f32 + f33*f33;
    *energy = volumes[index] * g * .5 * (sq_sum - 3 - 2*(1 + excess_osm)*log(det));
    *energy += volumes[index] * kt_o_omega * ((det - 1)*log((det - 1) / det) + chi*(det-1)/det);
}

double parallel_calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi){

    vector<double> energies(tetrahedra.size() / 4, 0);

    //Compute energies in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tetrahedra.size()/4),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                gel_energy_worker(pts, tetrahedra, inv_mats, volumes, g, excess_osm, kt_o_omega, chi, index, &energies[index]);
            }
        }
    );

    //Sum energies using a parallel reduction
    return sum_reduce(&energies[0], &energies[0]+energies.size(), 0.);
}

//Given a rubber modulus, a temperature over a solvent volume, a Flory-Huggins
//chi parameter, and a network bulk modulus, find the equilibrium swelling 
//ratio of a composite tissue, and return the corresponding principal stretch.
//double calc_equilibrium_swelling(double g, double excess_osm, double kt_o_omega, double chi, double k_network, int max_steps){
double calc_equilibrium_swelling(double g, double excess_osm, double kt_o_omega, double chi, int max_steps){

    //Relative dilation and first and second derivatives.
    double j, first_deriv, second_deriv;

    int steps = 0;
    j = 2;
    //first_deriv = g*(pow(j, -1./3) - (1+excess_osm)/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j)) + k_network*(j-1);
    //first_deriv = g*(pow(j, -1./3) - 1/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j)) + k_network/3*(pow(j, -1./3) - pow(j, -2./3));
    first_deriv = g*(pow(j, -1./3) - 1/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j));

    while(abs(first_deriv) > TOL && steps < max_steps){
        //second_deriv = g*((1+excess_osm)/j/j - pow(j,-4./3)/3) + kt_o_omega*(1/(j*(j-1)) - 1/j/j - 2*chi/(j*j*j)) + k_network;
        //second_deriv = g*(1/j/j - pow(j,-4./3)/3) + kt_o_omega*(1/(j*(j-1)) - 1/j/j - 2*chi/(j*j*j)) + k_network/9*(2*pow(j,-5./3) - pow(j, -4./3));
        second_deriv = g*(1/j/j - pow(j,-4./3)/3) + kt_o_omega*(1/(j*(j-1)) - 1/j/j - 2*chi/(j*j*j));
        j = max(j - first_deriv / second_deriv, 1 + TOL);
        //first_deriv = g*(pow(j, -1./3) - (1+excess_osm)/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j)) + k_network*(j-1);
        //first_deriv = g*(pow(j, -1./3) - 1/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j)) + k_network/3*(pow(j, -1./3) - pow(j, -2./3));
        first_deriv = g*(pow(j, -1./3) - 1/j) + kt_o_omega*(log((j-1)/j)+1/j+chi/(j*j));
        steps ++;
    }

    //cout << "Final first derivative: " << first_deriv << endl;

    return pow(j, 1./3.);
}

//Uniformly dilate a composite tissue in a serial manner
void dilate_vertices(vector<double> &points, vector<double> &bounds, double coeff){

    int index;

    for(index = 0; index < points.size() - 2; index += 3){
        points[index] = bounds[0] + (points[index] - bounds[0])*coeff;
        points[index + 1] = bounds[2] + (points[index + 1] - bounds[2])*coeff;
        points[index + 2] = bounds[4] + (points[index + 2] - bounds[4])*coeff;
    }
}

//Uniformly dilate a composite tissue in a parallel manner.
void dilate_vertices_parallel(vector<double> &points, vector<double> &bounds, double coeff){

    tbb::parallel_for( tbb::blocked_range<size_t>(0, points.size() / 3),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                points[3*index] = bounds[0] + (points[3*index] - bounds[0])*coeff;
                points[3*index+1] = bounds[2] + (points[3*index + 1] - bounds[2])*coeff;
                points[3*index+2] = bounds[4] + (points[3*index + 2] - bounds[4])*coeff;
            }
        }
    );
}

//Affinely deform a tissue by specifying a displacement along the top while
//keeping the bottom fixed, in a serial manner.
void affine_deform(vector<double> &points, vector<double> &bounds, Vec3D disp){

    int index;
    double range = bounds[5] - bounds[4], factor;

    /*cout << "Z displacement: " << disp.z << endl;
    cout << "Z minimum: " << bounds[4] << endl;*/

    //Displace coordinates by vectors proportional to their heights
    for(index = 0; index < points.size() - 2; index += 3){
        factor = (points[index+2] - bounds[4]) / range;
        points[index] += disp.x * factor;
        points[index+1] += disp.y * factor;
        points[index+2] += disp.z * factor;
    }

    //Update upper coordinate bounds
    bounds[1] += disp.x * (bounds[5] - bounds[4]);
    bounds[3] += disp.y * (bounds[5] - bounds[4]);
    bounds[5] += disp.z * (bounds[5] - bounds[4]);
}

//Affinely deform a tissue by specifying a displacement along the top while
//keeping the bottom fixed, in a parallel manner.
void affine_deform_parallel(vector<double> &points, vector<double> &bounds, Vec3D disp){

    double range = bounds[5] - bounds[4];

    //Displace coordinates by vectors proportional to their heights
    tbb::parallel_for( tbb::blocked_range<size_t>(0, points.size() / 3),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                double factor = (points[3*index+2] - bounds[4]) / range;
                points[3*index] += disp.x * factor;
                points[3*index+1] += disp.y * factor;
                points[3*index+2] += disp.z * factor;
            }
        }
    );

    bounds[1] += disp.x * (bounds[5] - bounds[4]);
    bounds[3] += disp.y * (bounds[5] - bounds[4]);
    bounds[5] += disp.z * (bounds[5] - bounds[4]);
}

//Helper function for computation of restoring forces
void compute_forces(vector<double> points, unordered_set<int> top, unordered_set<int> bottom, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats,  double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces){

    cblas_dscal(forces.size(), 0, &forces[0], 1);

    //time_t curr, prev;

    //time(&prev);

    if(ks > 0){
        stretching_forces(points, edges, lengths, ks, forces, 0,edges.size()/2);
    }
    //cout << "Force magnitude after stretching: " << cblas_dnrm2(forces.size(), &forces[0], 1) << endl;

    /*time(&curr);
    cout << "Stretching force time: " << difftime(curr, prev) << endl;
    time(&prev);*/

    if(kb > 0){
        bending_forces(points, triples, angles, kb, forces, 0,triples.size()/3);
    }
    //cout << "Force magnitude after bending: " << cblas_dnrm2(forces.size(), &forces[0], 1) << endl;
    /*time(&curr);
    cout << "Bending force time: " << difftime(curr, prev) << endl;
    time(&prev);*/

    gel_forces(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi, forces, 0, tets.size() / 4);
    //cout << "Force magnitude after gel forces: " << cblas_dnrm2(forces.size(), &forces[0], 1) << endl;
    /*time(&curr);
    cout << "Gel force time: " << difftime(curr, prev) << endl;*/

    for(int next_idx : bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
}

//Perform a structural relaxation of a composite tissue in a serial manner.
int nlcg_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, NLCGParams cgpar, double &f_rms){

    int num_dof, cg_iter = 0, cg_max, ls_iter, ls_max, reset_counter = 0;
    vector<double> forces, search_dir, prev_forces;
    double ks, kb, gmod, delta, kt_o_omega, chi, d_curr, d_orig;
    double d_mid, d_prev, ls_coeff, dir_coeff, cg_tol_sqr, sd_mag_sqr;
    double grad_dir_prod, prev_grad_dir_prod;

    num_dof = points.size();
    forces = vector<double>(num_dof, 0);
    //adv_forces = vector<double>(num_dof, 0);
    search_dir = vector<double>(num_dof, 0);
    //adv_pos = vector<double>(num_dof, 0);
    prev_forces = vector<double>(num_dof, 0);

    //Unpack mechanical parameters
    ks = mech_params[0];
    kb = mech_params[1];
    gmod = mech_params[2];
    delta = mech_params[3];
    kt_o_omega = mech_params[4];
    chi = mech_params[5];

    //Initialize nonlinear conjugate gradient parameters
    cg_max = cgpar.max_cg_iter;
    ls_max = cgpar.max_ls_iter;
    cg_tol_sqr = cgpar.tol*cgpar.tol;

    compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);

    d_curr = cblas_dnrm2(forces.size(), &forces[0], 1);
    //cout << "Force magnitude: " << d_curr << endl;
    d_orig = d_curr;
    cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
    cblas_dcopy(num_dof, &forces[0], 1, &prev_forces[0], 1);

    do {

        cg_iter ++;

        sd_mag_sqr = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);

        //cout << "Search direction magnitude squared: " << sd_mag_sqr << endl;

        //Compute the inner product of the force a small distance along the
        //search direction with the search direction itself to facilitate the
        //secant line search method.
        cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
        compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
        prev_grad_dir_prod = -cblas_ddot(num_dof,&forces[0],1,&search_dir[0],1);
        cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
        cblas_dcopy(num_dof, &prev_forces[0], 1, &forces[0], 1);
        ls_coeff = -FINITE_DIFF_COEFF;

        //Carry out a Newton's Method line search
	    ls_iter = 0;
	    do {

	        //Forces a short distance away along the current search direction
	        //cblas_dcopy(num_dof, &points[0], 1, &adv_pos[0], 1);
            //cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &adv_pos[0], 1);
            //compute_forces(adv_pos, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, adv_forces);
            //cout << "Advanced force magnitude: " << cblas_dnrm2(adv_forces.size(), &adv_forces[0], 1) << endl;

            //cblas_daxpy(num_dof, -1, &forces[0], 1, &adv_forces[0], 1);
            //cblas_dscal(num_dof, 1 / FINITE_DIFF_COEFF, &adv_forces[0], 1);
            //ls_coeff = -cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
            //cout << "Initial line search coefficient: " << ls_coeff << endl;
            //ls_coeff /= cblas_ddot(num_dof, &search_dir[0], 1, &adv_forces[0], 1);
            grad_dir_prod = -cblas_ddot(num_dof,&forces[0],1,&search_dir[0],1);
            ls_coeff *= grad_dir_prod / (prev_grad_dir_prod - grad_dir_prod);
            //cout << "Line search coefficient: " << ls_coeff << endl;
            //if(ls_coeff < 0 && ls_iter > 0) break;
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &points[0], 1);

            compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
            //cout << "Updated force magnitude: " << cblas_dnrm2(forces.size(), &forces[0], 1) << endl;
	        ls_iter ++;
            prev_grad_dir_prod = grad_dir_prod;
        } while(ls_iter < ls_max && sd_mag_sqr*ls_coeff*ls_coeff > cg_tol_sqr);

        //cout << "Line search iterations: " << ls_iter << endl;

        d_prev = d_curr;
	    d_curr = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);

        reset_counter ++;

        //Periodically reset the search direction, as information about past
        //CG iterations should periodically be discarded.
        if(reset_counter == num_dof){
            reset_counter = 0;
            cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
        }

        else{
            d_mid = cblas_ddot(num_dof, &forces[0], 1, &prev_forces[0], 1);
            dir_coeff = max((d_curr - d_mid) / d_prev, 0);
            //cout << "Direction coefficient: " << dir_coeff << endl;
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1, &forces[0], 1, &search_dir[0], 1);
        }

        cblas_dcopy(num_dof, &forces[0], 1, &prev_forces[0], 1);
    } while(cg_iter < cg_max && d_curr > cg_tol_sqr*d_orig);

    f_rms = cblas_dnrm2(forces.size(), &forces[0], 1) / sqrt(num_dof);
    return cg_iter;
}

int cgtr_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, CGTRParams cgtrpar, double &f_rms){

    int num_dof, max_outer, max_inner, n_outer, n_inner;
    double max_radius, min_ratio, conv_tol_sqr, radius, cf_model_ratio, curr_cf;
    double prev_cf, curr_mod, prev_mod, sd_hess_norm, ls_coeff, fmag_sqr;
    double fmag_sqr_0, ks, kb, gmod, delta, kt_o_omega, chi, dir_coeff, a, b, c;
    double inner_tol_sqr, rmag_sqr, prev_rmag_sqr, mod_val;
    double prev_mod_val, root_1, root_2, mod_val_1, mod_val_2;
    vector<double> forces, search_dir, hess_dir_proj, step;//, prev_r;

    num_dof = points.size();

    //Unpack relaxation parameters and physical parameters
    max_outer = cgtrpar.max_outer;    
    max_inner = cgtrpar.max_inner;
    max_radius = cgtrpar.max_radius;
    min_ratio = cgtrpar.min_ratio;
    conv_tol_sqr = cgtrpar.conv_tol*cgtrpar.conv_tol; 

    ks = mech_params[0];
    kb = mech_params[1];
    gmod = mech_params[2];
    delta = mech_params[3];
    kt_o_omega = mech_params[4];
    chi = mech_params[5];

    forces = vector<double>(num_dof, 0);
    search_dir = vector<double>(num_dof, 0);
    hess_dir_proj = vector<double>(num_dof, 0);
    step = vector<double>(num_dof, 0);
    //prev_r = vector<double>(num_dof, 0);

    //Evaluate the energy and forces at the outset
    compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
    fmag_sqr_0 = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);
    fmag_sqr = fmag_sqr_0;
    radius = max_radius;

    if(edges.size()){
        prev_cf = stretching_energy(points, edges, lengths, ks);
    }
    else prev_cf = 0;
    if(triples.size()){
        prev_cf += fiber_bending_energy(points, triples, angles, kb);
    }
    if(tets.size()){
        prev_cf += calc_gel_energy(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi);
    }

    for(n_outer = 0; n_outer < max_outer; ++n_outer){

        cblas_dscal(num_dof, 0, &step[0], 1);
        rmag_sqr = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);
        prev_rmag_sqr = rmag_sqr;
        cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
        //cblas_dcopy(num_dof, &forces[0], 1, &prev_r[0], 1);
        //cblas_dscal(num_dof, -1, &prev_r[0], 1);
        inner_tol_sqr = min(.25, prev_rmag_sqr)*prev_rmag_sqr;
        prev_mod_val = prev_cf;

        //Seek an approximate optimal point for the local model within the
        //current trust region.
        n_inner = 0;
        do if(rmag_sqr > inner_tol_sqr){

            //Find the curvature along the present search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
            compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj);
            cblas_daxpy(num_dof, -1., &forces[0], 1, &hess_dir_proj[0], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[0], 1);
            sd_hess_norm = cblas_ddot(num_dof, &search_dir[0], 1, &hess_dir_proj[0], 1); 
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);

            //If the curvature along the search direction is zero or positive,
            //proceed out to the trust region boundary.
            if(sd_hess_norm <= 0){
                a = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);
                b = cblas_ddot(num_dof, &search_dir[0], 1, &step[0], 1);
                c = cblas_ddot(num_dof,&step[0],1,&step[0],1) - radius*radius;
                root_1 = (sqrt(b*b + a*c) - b) / a;
                root_2 = -2.*b - 2.*root_1;

                //Evaluate the model for both steps, and choose the root that
                //minimizes the local quadratic approximation.
                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[0], 1);
                mod_val_2 += .5*root_1*root_1*sd_hess_norm;

                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[0], 1);
                mod_val_2 += .5*root_1*root_1*sd_hess_norm;

                if(mod_val_1 < mod_val_2){
                    cblas_daxpy(num_dof, root_1, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_1 + prev_mod_val;
                }
                else{
                    cblas_daxpy(num_dof, root_2, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_2 + prev_mod_val;
                }
                cblas_daxpy(num_dof, 1, &step[0], 1, &points[0], 1);
                break;
            }

            //If the curvature is positive, attempt a Newton-style update step.
            //If the update step leaves the trust region, truncate the step
            //at the boundary.
            ls_coeff = rmag_sqr / sd_hess_norm;
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &step[0], 1);

            if(cblas_ddot(num_dof, &step[0], 1, &step[0], 1) > radius*radius){
                a = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);
                b = cblas_ddot(num_dof, &search_dir[0], 1, &step[0], 1);
                c = cblas_ddot(num_dof,&step[0],1,&step[0],1) - radius*radius;
                root_1 = (sqrt(b*b + a*c) - b) / a;
                cblas_daxpy(num_dof, root_1-ls_coeff, &step[0], 1, &step[0], 1);
                break;
            }

            //Evaluate the new model residual and update the search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &step[0], 1, &points[0], 1);
            compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj);
            cblas_daxpy(num_dof, -1., &forces[0], 1, &hess_dir_proj[0], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[0], 1);
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &step[0], 1, &points[0],1);

            prev_rmag_sqr = rmag_sqr;
            //rold_rnew = cblas_ddot(num_dof, &prev_r[0], 1,&hess_dir_proj[0], 1);
            //rold_rnew -= cblas_ddot(num_dof, &prev_r[0], 1, &forces[0], 1);
            rmag_sqr = fmag_sqr + cblas_ddot(num_dof, &hess_dir_proj[0], 1, &hess_dir_proj[0], 1) - 2*cblas_ddot(num_dof, &forces[0], 1, &hess_dir_proj[0], 1);

            if(rmag_sqr <= inner_tol_sqr) break;

            dir_coeff = rmag_sqr / prev_rmag_sqr;
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1., &forces[0], 1, &search_dir[0], 1); 
            cblas_daxpy(num_dof, -1., &hess_dir_proj[0], 1, &search_dir[0], 1); 
            //cblas_dcopy(num_dof, &hess_dir_proj[0], 1, &prev_r[0], 1);
            //cblas_daxpy(num_dof, -1., &forces[0], 1, &prev_r[0], 1); 
            n_inner ++;
        } while(n_inner < max_inner);

        //Evaluate the energy after a step is proposed, and compare the actual
        //and expected reduction. Modify the trust radius as appropriate, and
        //decide whether to keep the search step.
        cblas_daxpy(num_dof, 1., &step[0], 1, &points[0], 1);
        if(edges.size()){
            curr_cf = stretching_energy(points, edges, lengths, ks);
        }
        else curr_cf = 0;
        if(triples.size()){
            curr_cf += fiber_bending_energy(points, triples, angles, kb);
        }
        if(tets.size()){
            curr_cf += calc_gel_energy(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi);
        }

        cf_model_ratio = (prev_cf - curr_cf) / (prev_cf - mod_val); 

        //Address poor performance of the model by shrinking the trust radius
        if(cf_model_ratio < .25){
            radius /= 4.;
        }

        else{
            //If the model performs well and the search reaches the trust region
            //boundary, expand the trust region.
            if(cf_model_ratio > .75 && abs(cblas_ddot(num_dof, &step[0], 1, &step[0], 1) - radius*radius) < TOL){
                radius = min(2*radius, max_radius);
            }
        }

        //If the ratio of actual to predicted improvement is inadequate, reject
        //the computed update step.
        if(cf_model_ratio < min_ratio){
            compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
        }
        else cblas_daxpy(num_dof, -1., &step[0], 1, &points[0], 1);

        prev_cf = curr_cf;
        fmag_sqr = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);
        if(fmag_sqr / fmag_sqr_0 < conv_tol_sqr) break;

        //cout << "Iteration " << n_outer << endl;
    }

    if(fmag_sqr / fmag_sqr_0 < conv_tol_sqr){
        f_rms = cblas_dnrm2(forces.size(), &forces[0], 1) / sqrt(num_dof);
        return n_outer;
    }

    //Refine the global search with a local nonlinear CG search
    NLCGParams cgpar(max_outer, max_inner, cgtrpar.conv_tol*sqrt(fmag_sqr_0/fmag_sqr));

    n_outer += nlcg_relax(points, bottom, top, edges, lengths, triples, angles, tets, volumes, inv_mats, mech_params, cgpar, f_rms);

    return n_outer;
}

//Helper function to handle parallel force computation
void compute_forces_parallel(vector<double> points, unordered_set<int> top, unordered_set<int> bottom, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments){

    int iter, num_dof;
    //time_t curr, prev;

    num_dof = forces.size();

    cblas_dscal(num_dof, 0, &forces[0], 1);
    for(iter = 0; iter < force_sets.size(); iter++){
        cblas_dscal(num_dof, 0, &force_sets[iter][0], 1);
    }

    tbb::task_group g;

    //time(&prev);

    if(ks > 0){
        for(iter = 0; iter < stretch_assignments.size() - 1; iter++){
            g.run([points, edges, lengths, ks, &force_sets, stretch_assignments, iter]{
                stretching_forces(points, edges, lengths, ks, force_sets[iter], stretch_assignments[iter], stretch_assignments[iter+1]);
            });
        }
        g.wait();
    }

    /*time(&curr);
    cout << "Stretching force time: " << difftime(curr, prev) << endl;
    time(&prev);*/

    if(kb > 0){
        for(iter = 0; iter < bend_assignments.size() - 1; iter++){
            g.run([points, triples, angles, kb, &force_sets, bend_assignments, iter]{
                bending_forces(points, triples, angles, kb, force_sets[iter], bend_assignments[iter], bend_assignments[iter+1]);
            });
        }
        g.wait();
    }

    /*time(&curr);
    cout << "Bending force time: " << difftime(curr, prev) << endl;
    time(&prev);*/

    if(tets.size()){
        for(iter = 0; iter < gel_assignments.size() - 1; iter++){
            g.run([points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi, &force_sets, gel_assignments, iter]{
                gel_forces(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi, force_sets[iter], gel_assignments[iter], gel_assignments[iter+1]);
            });
        }
        g.wait();
    }

    /*time(&curr);
    cout << "Gel force time: " << difftime(curr, prev) << endl;
    time(&curr);*/

    for(iter = 0; iter < force_sets.size(); iter++){
        cblas_daxpy(num_dof, 1., &force_sets[iter][0], 1, &forces[0], 1);
    }

    for(int next_idx : bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }

    /*time(&curr);
    cout << "Summation time: " << difftime(curr, prev) << endl;*/
}

//Perform a structural relaxation of a composite tissue in a parallel manner.
int parallel_nlcg_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, NLCGParams cgpar, int n_threads, double &f_rms){

    int num_dof, cg_iter = 0, cg_max, ls_iter, ls_max, reset_counter = 0;
    vector<double> forces, search_dir, prev_forces;
    double ks, kb, gmod, delta, kt_o_omega, chi, d_curr, d_orig;
    double d_mid, d_prev, ls_coeff, dir_coeff, cg_tol_sqr, sd_mag_sqr;
    double grad_dir_prod, prev_grad_dir_prod;
    vector<vector<double>> force_sets;
    vector<int> stretch_assignments, bend_assignments, gel_assignments;

    num_dof = points.size();

    //Allocate sets of auxilliary force vectors to enable parallel computation
    for(int iter = 0; iter < n_threads; iter++){
        force_sets.push_back(vector<double>(num_dof, 0));
    }

    //Determine how jobs should be allotted to threads for each type of force
    //to be computed.
    stretch_assignments = job_assignments(edges.size()/2, n_threads);
    bend_assignments = job_assignments(triples.size()/3, n_threads);
    gel_assignments = job_assignments(tets.size()/4, n_threads);

    num_dof = points.size();
    forces = vector<double>(num_dof, 0);
    //adv_forces = vector<double>(num_dof, 0);
    search_dir = vector<double>(num_dof, 0);
    //adv_pos = vector<double>(num_dof, 0);
    prev_forces = vector<double>(num_dof, 0);

    //Unpack mechanical parameters
    ks = mech_params[0];
    kb = mech_params[1];
    gmod = mech_params[2];
    delta = mech_params[3];
    kt_o_omega = mech_params[4];
    chi = mech_params[5];

    //Initialize nonlinear conjugate gradient parameters
    cg_max = cgpar.max_cg_iter;
    ls_max = cgpar.max_ls_iter;
    cg_tol_sqr = cgpar.tol*cgpar.tol;

    compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);

    d_curr = cblas_dnrm2(forces.size(), &forces[0], 1);
    d_orig = d_curr;
    cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
    cblas_dcopy(num_dof, &forces[0], 1, &prev_forces[0], 1);

    do {

        cg_iter ++;

        sd_mag_sqr = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);

        //Compute the inner product of the force with the search direction
        //to facilitate the secant method
        cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
        compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
        prev_grad_dir_prod = -cblas_ddot(num_dof, &forces[0], 1, &search_dir[0],1);
        cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
        cblas_dcopy(num_dof, &prev_forces[0], 1, &forces[0], 1);
        ls_coeff = -FINITE_DIFF_COEFF;


        //Carry out a Newton's Method line search
        ls_iter = 0;
        do {

            //Forces a short distance away along the current search direction
	        //cblas_dcopy(num_dof, &points[0], 1, &adv_pos[0], 1);
            //cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &adv_pos[0], 1);
            //compute_forces_parallel(adv_pos, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_records);

            //cblas_daxpy(num_dof, -1, &forces[0], 1, &adv_forces[0], 1);
            //cblas_dscal(num_dof, 1 / FINITE_DIFF_COEFF, &adv_forces[0], 1);
            //ls_coeff = -cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
            //ls_coeff /= cblas_ddot(num_dof, &search_dir[0], 1, &adv_forces[0], 1);
            grad_dir_prod = -cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
            ls_coeff *= grad_dir_prod / (prev_grad_dir_prod - grad_dir_prod);
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &points[0], 1);
            prev_grad_dir_prod = grad_dir_prod;
            ls_iter ++;

            //Forces at the current position
            compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);

        } while(ls_iter < ls_max && sd_mag_sqr*ls_coeff*ls_coeff > cg_tol_sqr);

        d_prev = d_curr;
        d_curr = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);

        reset_counter ++;
        //Periodically reset the search direction, as information about past
        //CG iterations should periodically be discarded.
        if(reset_counter == num_dof){
            reset_counter = 0;
	        cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
        }

        else{
            d_mid = cblas_ddot(num_dof, &forces[0], 1, &prev_forces[0], 1);
            dir_coeff = max((d_curr - d_mid) / d_prev, 0);
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1, &forces[0], 1, &search_dir[0], 1);
        }

        cblas_dcopy(num_dof, &forces[0], 1, &prev_forces[0], 1);
    } while(cg_iter < cg_max && d_curr > cg_tol_sqr*d_orig);

    f_rms = cblas_dnrm2(forces.size(), &forces[0], 1) / sqrt(num_dof);
    return cg_iter;
}

int parallel_cgtr_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, CGTRParams cgtrpar, int n_threads, double &f_rms){

    int num_dof, max_outer, max_inner, n_outer, n_inner;
    double max_radius, min_ratio, conv_tol_sq, radius, cf_model_ratio, curr_cf;
    double prev_cf, curr_mod, prev_mod, sd_hess_norm, ls_coeff, fmag_sqr;
    double fmag_sqr_0, ks, kb, gmod, delta, kt_o_omega, chi, dir_coeff, a, b, c;
    double inner_tol_sqr, rmag_sqr, prev_rmag_sqr, mod_val, prev_mod_val;
    double root_1, root_2, mod_val_1, mod_val_2;
    vector<double> forces, search_dir, hess_dir_proj, step;
    vector<vector<double>> force_sets;
    vector<int> stretch_assignments, bend_assignments, gel_assignments;

    num_dof = points.size();

    //Allocate sets of auxilliary force vectors to enable parallel computation
    for(int iter = 0; iter < n_threads; iter++){
        force_sets.push_back(vector<double>(num_dof, 0));
    }
    //Determine how jobs should be allotted to threads for each type of force
    //to be computed.
    stretch_assignments = job_assignments(edges.size()/2, n_threads);
    bend_assignments = job_assignments(triples.size()/3, n_threads);
    gel_assignments = job_assignments(tets.size()/4, n_threads);

    //Unpack relaxation parameters and physical parameters
    max_outer = cgtrpar.max_outer;    
    max_inner = cgtrpar.max_inner;
    max_radius = cgtrpar.max_radius;
    min_ratio = cgtrpar.min_ratio;
    conv_tol_sq = cgtrpar.conv_tol*cgtrpar.conv_tol; 

    ks = mech_params[0];
    kb = mech_params[1];
    gmod = mech_params[2];
    delta = mech_params[3];
    kt_o_omega = mech_params[4];
    chi = mech_params[5];

    forces = vector<double>(num_dof, 0);
    search_dir = vector<double>(num_dof, 0);
    hess_dir_proj = vector<double>(num_dof, 0);
    step = vector<double>(num_dof, 0);

    //Evaluate the energy and forces at the outset
    compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
    fmag_sqr_0 = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);
    fmag_sqr = fmag_sqr_0;
    radius = max_radius;

    if(edges.size()){
        prev_cf = stretching_energy(points, edges, lengths, ks);
    }
    else prev_cf = 0;
    if(triples.size()){
        prev_cf += fiber_bending_energy(points, triples, angles, kb);
    }
    if(tets.size()){
        prev_cf += calc_gel_energy(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi);
    }

    for(n_outer = 0; n_outer < max_outer; ++n_outer){

        cblas_dscal(num_dof, 0, &step[0], 1);
        rmag_sqr = cblas_ddot(num_dof, &forces[0], 1, &forces[0], 1);
        prev_rmag_sqr = rmag_sqr;
        cblas_dcopy(num_dof, &forces[0], 1, &search_dir[0], 1);
        inner_tol_sqr = min(.25, prev_rmag_sqr)*prev_rmag_sqr;
        prev_mod_val = prev_cf;

        //Seek an approximate optimal point for the local model within the
        //current trust region.
        for(n_inner = 0; n_inner < max_inner; ++n_inner){

            //Find the curvature along the present search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);
            compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            cblas_daxpy(num_dof, -1., &forces[0], 1, &hess_dir_proj[0], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[0], 1);
            sd_hess_norm = cblas_ddot(num_dof, &search_dir[0], 1, &hess_dir_proj[0], 1); 
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &points[0], 1);

            //If the curvature along the search direction is zero or positive,
            //proceed out to the trust region boundary.
            if(sd_hess_norm <= 0){
                a = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);
                b = cblas_ddot(num_dof, &search_dir[0], 1, &step[0], 1);
                c = cblas_ddot(num_dof,&step[0],1,&step[0],1) - radius*radius;
                root_1 = (sqrt(b*b + a*c) - b) / a;
                root_2 = -2.*b - 2.*root_1;

                //Evaluate the model for both steps, and choose the root that
                //minimizes the local quadratic approximation.
                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[0], 1);
                mod_val_2 += .5*root_1*root_1*sd_hess_norm;

                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[0], 1);
                mod_val_2 += .5*root_1*root_1*sd_hess_norm;

                if(mod_val_1 < mod_val_2){
                    cblas_daxpy(num_dof, root_1, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_1 + prev_mod_val;
                }
                else{
                    cblas_daxpy(num_dof, root_2, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_2 + prev_mod_val;
                }
                cblas_daxpy(num_dof, 1, &step[0], 1, &points[0], 1);
                break;
            }

            //If the curvature is positive, attempt a Newton-style update step.
            //If the update step leaves the trust region, truncate the step
            //at the boundary.
            ls_coeff = rmag_sqr / sd_hess_norm;
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &step[0], 1);

            if(cblas_ddot(num_dof, &step[0], 1, &step[0], 1) > radius*radius){
                a = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);
                b = cblas_ddot(num_dof, &search_dir[0], 1, &step[0], 1);
                c = cblas_ddot(num_dof,&step[0],1,&step[0],1) - radius*radius;
                root_1 = (sqrt(b*b + a*c) - b) / a;
                cblas_daxpy(num_dof, root_1-ls_coeff, &step[0], 1, &step[0], 1);
                break;
            }

            //Evaluate the new model residual and update the search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &step[0], 1, &points[0], 1);
            compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            cblas_daxpy(num_dof, -1., &forces[0], 1, &hess_dir_proj[0], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[0], 1);
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &step[0], 1, &points[0],1);

            prev_rmag_sqr = rmag_sqr;
            rmag_sqr = fmag_sqr + cblas_ddot(num_dof, &hess_dir_proj[0], 1, &hess_dir_proj[0], 1) - 2*cblas_ddot(num_dof, &forces[0], 1, &hess_dir_proj[0], 1);

            if(rmag_sqr <= inner_tol_sqr) break;

            dir_coeff = rmag_sqr / prev_rmag_sqr;
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1., &forces[0], 1, &search_dir[0], 1); 
            cblas_daxpy(num_dof, -1., &hess_dir_proj[0], 1, &search_dir[0], 1); 
        }

        //Evaluate the energy after a step is proposed, and compare the actual
        //and expected reduction. Modify the trust radius as appropriate, and
        //decide whether to keep the search step.
        cblas_daxpy(num_dof, 1., &step[0], 1, &points[0], 1);
        if(edges.size()){
            curr_cf = stretching_energy(points, edges, lengths, ks);
        }
        else curr_cf = 0;
        if(triples.size()){
            curr_cf += fiber_bending_energy(points, triples, angles, kb);
        }
        if(tets.size()){
            curr_cf += calc_gel_energy(points, tets, inv_mats, volumes, gmod, delta, kt_o_omega, chi);
        }

        cf_model_ratio = (prev_cf - curr_cf) / (prev_cf - mod_val); 

        //Address poor performance of the model by shrinking the trust radius
        if(cf_model_ratio < .25){
            radius /= 4.;
        }

        else{
            //If the model performs well and the search reaches the trust region
            //boundary, expand the trust region.
            if(cf_model_ratio > .75 && abs(cblas_ddot(num_dof, &step[0], 1, &step[0], 1) - radius*radius) < TOL){
                radius = min(2*radius, max_radius);
            }
        }

        //If the ratio of actual to predicted improvement is inadequate, reject
        //the computed update step.
        if(cf_model_ratio < min_ratio){
            compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
        }
        else cblas_daxpy(num_dof, -1., &step[0], 1, &points[0], 1);

        prev_cf = curr_cf;

        //cout << "Iteration " << n_outer << endl;
    }

    if(fmag_sqr / fmag_sqr_0 < conv_tol_sq){
        f_rms = cblas_dnrm2(forces.size(), &forces[0], 1) / sqrt(num_dof);
        return n_outer;
    }

    //Refine the global search with a local nonlinear CG search
    NLCGParams cgpar(max_outer, max_inner, cgtrpar.conv_tol*sqrt(fmag_sqr_0/fmag_sqr));

    n_outer += parallel_nlcg_relax(points, bottom, top, edges, lengths, triples, angles, tets, volumes, inv_mats, mech_params, cgpar, n_threads, f_rms);

    f_rms = cblas_dnrm2(forces.size(), &forces[0], 1) / sqrt(num_dof);
    return n_outer;
}

//Report the results of energy relaxation to a file
void report_edata(FILE *edat_file, Vec3D disp, double es, double eb, double eg){

    fprintf(edat_file, "%2.10lf\t%2.10f\t%2.10lf\t%1.10le\t%1.10le\t%1.10le\t%1.10le\n", disp.x, disp.y, disp.z, es, eb, eg, es+eb+eg);
}

//Report data about the quality of convergence, the number of relaxation
//iterations, and the time required
void log_entry(FILE *log_file, double f_rms, int steps, clock_t ct){

    fprintf(log_file, "RMS Force: %1.10le Steps: %d Elapsed time: %lf seconds\n", f_rms, steps, ((double) ct) / CLOCKS_PER_SEC);
}

//Direct a course of deformation and structural relaxation. Given the specified
//number of threads, use either the serial or parallel structural relaxation
//function. Report the relaxed energy, deformed coordinates, and logfile
//information for each coarse increment.
void do_relaxation_run(string nmc_file, string log_name, string edat_name, string nmt_base, vector<double> mech_params, CGTRParams par, vector<DisplacementProgram> progs, bool strain_mode, int n_threads){

    int n_steps, num_digits, i, j;
    double dilation, volume, estretch, ebend, egel, k_network, f_rms, swelling;
    vector<double> bounds, points, lengths, angles, volumes, inv_mats;
    unordered_set<int> top, bottom;
    vector<int> edges, triples, elements;
    Vec3D total_displacement, displacement, coarse_step, fine_step;
    FILE *edat_file, *log_file;
    string xml_name, nmt_name;
    //Data type for timing the relaxation run
    clock_t ct;
    bool parallel = n_threads > 1;

    double minx, maxx, miny, maxy, minz, maxz;

    //Attempt to read a network mesh composite (NMC) file, and intialize
    //vertices, edges, triples, mesh elements, and associated data structures.
    if(! parse_nmc_file(nmc_file, bounds, points, bottom, top, edges, lengths, triples, angles, elements)){
        cerr << "Parsing of the NMC file failed.\n";
        return;
    }

    //Attempt to open a log file
    log_file = fopen(log_name.c_str(), "w");
    if(log_file == NULL){
        cerr << "The log file could not be opened.\n";
        return;
    }

    //Attempt to open the energy file
    edat_file = fopen(edat_name.c_str(), "w");
    if(edat_file == NULL){
        cerr << "The energy file could not be opened.\n";
        fclose(log_file);
        return;
    }

    //If detailed displaced position data are to be recorded, open a manifest
    if(nmt_base != ""){
        RelaxParams rp(mech_params[0], mech_params[1], mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
        if(!open_network_mesh_trajectory(nmt_base, nmc_file, rp, xml_name, nmt_name)){
            cerr << "The network-mesh trajectory files could not be opened.\n";
            fclose(log_file);
            fclose(edat_file);
            return;
        }
    }

    volume = (bounds[1]-bounds[0])*(bounds[3]-bounds[2])*(bounds[5]-bounds[4]);

    //Find the equilibrium swollen state at the outset.

    //Gague the network bulk modulus
    /*dilate_vertices(points, bounds, 1+TRIAL_D);
    if(! parallel){
        estretch = stretching_energy(points, edges, lengths, mech_params[0]);
	    ebend = fiber_bending_energy(points, triples, angles, mech_params[1]);
    }
    else{
        estretch = parallel_stretching_energy(points, edges, lengths, mech_params[0]);
	    ebend = parallel_fiber_bending_energy(points, triples, angles, mech_params[1]);
    }

    cout << "Stretching energy: " << estretch << endl;
    cout << "Bending energy: " << ebend << endl;
    dilation = (1.+TRIAL_D)*(1.+TRIAL_D)*(1.+TRIAL_D) - 1;
    k_network = 2*(estretch+ebend) / (volume * TRIAL_D*TRIAL_D);
    cout << "Network bulk modulus: " << k_network << endl;
    Return the network to its initial state
    dilate_vertices(points, bounds, 1/(1+TRIAL_D));
    */

    //Use root-finding to compute the swelling equilibrium, apply the
    //appropriate swelling ratio, and equilibrate to account for local
    //heterogeneity. 
    swelling = calc_equilibrium_swelling(mech_params[2], mech_params[3], mech_params[4], mech_params[5], DEFAULT_MAX_STEPS);
    cout << "Swelling: " << swelling << endl;
    dilate_vertices(points, bounds, 1/swelling);
    //Find tetrahedron volumes
    calc_volumes(points, elements, volumes);
    //Find matrices for computing deformation gradient tensors
    get_inv_mats(points, elements, inv_mats);
    dilate_vertices(points, bounds, swelling);
    total_displacement = Vec3D(0, 0, 0);
    estretch = stretching_energy(points, edges, lengths, mech_params[0]);
    cout << "Stretching energy after dilation: " << estretch << endl;
    egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    cout << "Gel energy: " << egel << endl;

    minx = FLT_MAX;
    maxx = -FLT_MAX;
    miny = FLT_MAX;
    maxy = -FLT_MAX;
    minz = FLT_MAX;
    maxz = -FLT_MAX;
    for(int pt_iter = 0; pt_iter < points.size(); pt_iter += 3){
        if(points[pt_iter] < minx) minx = points[pt_iter];
        if(points[pt_iter] > maxx) maxx = points[pt_iter];
        if(points[pt_iter+1] < miny) miny = points[pt_iter+1];
        if(points[pt_iter+1] > maxy) maxy = points[pt_iter+1];
        if(points[pt_iter+2] < minz) minz = points[pt_iter+2];
        if(points[pt_iter+2] > maxz) maxz = points[pt_iter+2];
    }
    cout << "Bounds at the start: (" << minx << ", " << maxx << "), (" << miny << ", " << maxy << "), (" << minz << ", " << maxz << ")\n";

    ct = clock();
    if(! parallel){
        n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
    }
    else {
        n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
    }
    ct = clock() - ct;

    if(! parallel){
        estretch = stretching_energy(points, edges, lengths, mech_params[0]);
	    ebend = fiber_bending_energy(points, triples, angles, mech_params[1]);
        egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    }
    else{
        estretch = parallel_stretching_energy(points, edges, lengths, mech_params[0]);
	    ebend = parallel_fiber_bending_energy(points, triples, angles, mech_params[1]);
	    egel = parallel_calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    }
    report_edata(edat_file, total_displacement, estretch, ebend, egel);
    log_entry(log_file, f_rms, n_steps, ct);

    //If position data are to be recorded, record the initial, swollen state of
    //the network.
    if(nmt_base != ""){
        if(!append_network_mesh_state(total_displacement, points, xml_name, nmt_name)){
            cerr << "Appending of position data failed.\n";
	        fclose(log_file);
            fclose(edat_file);
            return;
        }
    }

    //Iteratively apply a prescribed strain at the top, affinely deform the 
    //network, relax about the affine guess, and report the final configuration
    //of the system, the strain energy, and a record of the relaxation attempt.
    for(DisplacementProgram next_prog : progs){

        //Find the coarse increment. If displacements are specified in relative
        //terms, rescale displacements by the current sample height.
        coarse_step = next_prog.coarse_step;
        if(strain_mode){
            scalar_multiply(coarse_step, bounds[5] - bounds[4]);
        }

        //Find the fine increment, if a fine increment is to be used
        if(next_prog.num_fine > 0){
            fine_step = next_prog.fine_step;
            if(strain_mode){
                scalar_multiply(fine_step, bounds[5] - bounds[4]);
            }
        }

        //Advance the top of the network in a series of displacements. Affinely
        //displace the rest of the network by an amount proportional to the
        //height above the bottom of the sample. Relax about this affine guess,
        //and report displacements, itemized energy contributions and total
        //energy.
        copy(coarse_step, displacement);
        //If fine steps are desired, to aid in taking numerical derivatives,
        //add an offset to center a set of displacements about the
        //coarse displacement.
        if(next_prog.num_fine > 0){
            axpy(fine_step, displacement, -1. * next_prog.num_fine);
        }
        egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
        cout << "Gel energy before displacement: " << egel << endl;
        minx = FLT_MAX;
        maxx = -FLT_MAX;
        miny = FLT_MAX;
        maxy = -FLT_MAX;
        minz = FLT_MAX;
        maxz = -FLT_MAX;
        for(int pt_iter = 0; pt_iter < points.size(); pt_iter += 3){
            if(points[pt_iter] < minx) minx = points[pt_iter];
            if(points[pt_iter] > maxx) maxx = points[pt_iter];
            if(points[pt_iter+1] < miny) miny = points[pt_iter+1];
            if(points[pt_iter+1] > maxy) maxy = points[pt_iter+1];
            if(points[pt_iter+2] < minz) minz = points[pt_iter+2];
            if(points[pt_iter+2] > maxz) maxz = points[pt_iter+2];
        }
        cout << "Bounds before: (" << minx << ", " << maxx << "), (" << miny << ", " << maxy << "), (" << minz << ", " << maxz << ")\n";
        affine_deform(points, bounds, displacement);
        egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
        cout << "Gel energy after displacement: " << egel << endl;
        minx = FLT_MAX;
        maxx = -FLT_MAX;
        miny = FLT_MAX;
        maxy = -FLT_MAX;
        minz = FLT_MAX;
        maxz = -FLT_MAX;
        for(int pt_iter = 0; pt_iter < points.size(); pt_iter+=3){
            if(points[pt_iter] < minx) minx = points[pt_iter];
            if(points[pt_iter] > maxx) maxx = points[pt_iter];
            if(points[pt_iter+1] < miny) miny = points[pt_iter+1];
            if(points[pt_iter+1] > maxy) maxy = points[pt_iter+1];
            if(points[pt_iter+2] < minz) minz = points[pt_iter+2];
            if(points[pt_iter+2] > maxz) maxz = points[pt_iter+2];
        }
        cout << "Bounds after: (" << minx << ", " << maxx << "), (" << miny << ", " << maxy << "), (" << minz << ", " << maxz << ")\n";
        axpy(displacement, total_displacement, 1.);

        //The fine offset should be shortened for subsequent displacements
        if(next_prog.num_fine > 0){
            axpy(fine_step, displacement, -1. * next_prog.num_fine);
        }

        for(i = 0; i < next_prog.num_coarse; i++){
            if(next_prog.num_fine > 0){
                for(j = 0; j < 2*next_prog.num_fine + 1; j++){
                    ct = clock();
                    if(! parallel){
                        n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
                    }
                    else {
                        n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
                    }
                    ct = clock() - ct;
                    if(! parallel){
                        estretch = stretching_energy(points, edges, lengths, mech_params[0]);
                        ebend = fiber_bending_energy(points, triples, angles, mech_params[1]);
                        egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                    }
                    else{
                        estretch = parallel_stretching_energy(points, edges, lengths, mech_params[0]);
                        ebend = parallel_fiber_bending_energy(points, triples, angles, mech_params[1]);
                        egel = parallel_calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                    }
                    report_edata(edat_file, total_displacement, estretch, ebend, egel);
                    log_entry(log_file, f_rms, n_steps, ct);

                    //Only report displacements at coarse intervals
                    if(j == next_prog.num_fine){
                        if(nmt_base != ""){
                            if(!append_network_mesh_state(total_displacement, points, xml_name, nmt_name)){
                                cerr << "Appending of position data failed.\n";
                                fclose(log_file);
                                fclose(edat_file);
                                return;
                            }
                        }

                    }
                    if(j < 2*next_prog.num_fine){
                        affine_deform(points, bounds, fine_step);
			            axpy(fine_step, total_displacement, 1.);
                    }
                }

            }

            else{
                ct = clock();
                if(! parallel){
                    n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
                }
                else {
                    n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
                }
                ct = clock() - ct;

                if(! parallel){
                    estretch = stretching_energy(points, edges, lengths, mech_params[0]);
                    ebend = fiber_bending_energy(points, triples, angles, mech_params[1]);
                    egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                }
                else{
                    estretch = parallel_stretching_energy(points, edges, lengths, mech_params[0]);
                    ebend = parallel_fiber_bending_energy(points, triples, angles, mech_params[1]);
                    egel = parallel_calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                }
                report_edata(edat_file, total_displacement, estretch, ebend, egel);
                log_entry(log_file, f_rms, n_steps, ct);

    	        if(nmt_base != ""){
                    if(!append_network_mesh_state(total_displacement, points, xml_name, nmt_name)){
                        cerr << "Appending of position data failed.\n";
                        fclose(log_file);
                        fclose(edat_file);
                        return;
                    }
                }
            }

            if(i < next_prog.num_coarse - 1){
                affine_deform(points, bounds, displacement);
                axpy(displacement, total_displacement, 1.);
	        }
        }

    }

    //Finalize output files
    fclose(edat_file);
    fclose(log_file);
    if(nmt_base != ""){
        if(! close_network_mesh_trajectory(xml_name)){
            cerr << "Error finalizing position reporting.\n";
        }
    }
}

vector<double> parse_doubles(vector<string> numstring){
    double nextnum;
    vector<double> result;

    for(string s : numstring){
        if(sscanf(s.c_str(), "%lf", &nextnum)) result.push_back(nextnum);
    }

    return result;
}

bool parse_nlcg_params(string nextline, NLCGParams &cgpar){

    int cg_iter, ls_iter;
    double cg_tol;

    if(sscanf(nextline.c_str(), "%d %d %lf", &cg_iter, &ls_iter, &cg_tol) < 3){
        cerr << "Not all conjugate gradient parameters could  be read.\n";
        return false;
    }
    else if(cg_iter <= 0 || ls_iter <= 0 || cg_tol <= 0){
        cerr << "Not all conjugate gradient parameters were valid.\n";
        return false;
    }

    cgpar.max_cg_iter = cg_iter;
    cgpar.max_ls_iter = ls_iter;
    cgpar.tol = cg_tol;

    return true;
}

bool parse_cgtr_params(string nextline, CGTRParams &cgtrpar){

    int max_outer, max_inner;
    double max_radius, min_ratio, tol;

    if(sscanf(nextline.c_str(), "%d %d %lf %lf %lf", &max_outer, &max_inner, &max_radius, &min_ratio, &tol) < 5){
        cerr << "Not all Conjugate Gradient-Trust Region parameters could be read.\n";
        return false;
    }

    else if(max_outer <= 0 || max_inner <= 0 || max_radius <= 0 || min_ratio < 0 || tol <= 0){
        cerr << "Not all optimization parameters were valid.\n";
        return false;
    }

    cgtrpar.max_outer = max_outer;
    cgtrpar.max_inner = max_inner;
    cgtrpar.max_radius = max_radius;
    cgtrpar.min_ratio = min_ratio;
    cgtrpar.conv_tol = tol;

    return true;
} 

//Parse an input file specifying an input NMC file, a tag for labeling output,
//material properties, and a program of displacements to impose.
bool parse_input_file(string file_name, string &nmc_file, string &tag, vector<double> &mech_params, CGTRParams &cgtrpar, vector<DisplacementProgram> &progs){

    ifstream input;
    string nextline;
    vector<string> tokens;
    double x, y, z, fine_mag;
    int n_coarse, n_fine, num_read;
    Vec3D coarse_disp, fine_disp;

    input.open(file_name);
    if(! input.is_open()){
        cerr << "The input file could not be opened.\n";
        return false;
    }

    //Attempt to read NMC file and tag
    while(! input.eof()){
        getline(input, nextline);
        if(! nextline.compare("") || nextline[0] == '#') continue;

        tokens = split(nextline, ' ');
        nmc_file = tokens[0];
        if(tokens.size() > 1) tag = tokens[1];
        break;
    }
    if(input.eof()){
        input.close();
        cerr << "Not all information could be read from the job file.\n";
        return false;
    }

    //Attempt to read fiber stretching stiffness, fiber bending stiffness,
    //gel rubber modulus, excess osmolarity, temperature divided by solvent 
    //molecule volume, and chi parameter describing wetability
    while(! input.eof()){
        getline(input, nextline);
        if(! nextline.compare("") || nextline[0] == '#') continue;

        tokens = split(nextline, ' ');
        mech_params = parse_doubles(tokens);
        if(mech_params.size() != 6){
            cerr << "Six mechanical parameters could not be read.\n";
            input.close();
            return false;
        }
        break;
    }
    if(input.eof()){
        input.close();
        cerr << "Not all information could be read from the job file.\n";
        return false;
    }

    //Attempt to read instructions for relaxation. I have implemented both
    //the nonlinear conjugate gradient and Steihaug CG-Trust region methods.
    //I think a trust region will prove important in avoiding some  pathological
    //behavior I have observed.
    while(! input.eof()){
        getline(input, nextline);
        if(! nextline.compare("") || nextline[0] == '#') continue;
        if(! parse_cgtr_params(nextline, cgtrpar)){
            cerr << "Not all relaxation parameters could be read.\n";
            input.close();
            return false;
        }
        break;
    }
    if(input.eof()){
        input.close();
        cerr << "Not all information could be read from the job file.\n";
        return false;
    }

    //Attempt to read a series of deformations
    while(! input.eof()){
        getline(input, nextline);
        if(! nextline.compare("") || nextline[0] == '#') continue;
        num_read = sscanf(nextline.c_str(), "%lf %lf %lf %d %lf %d", &x, &y, &z, &n_coarse, &fine_mag, &n_fine);

        if(num_read == 4){
            progs.push_back(DisplacementProgram(Vec3D(x, y, z), n_coarse));
        }

        else if(num_read == 6){
            coarse_disp = Vec3D(x, y, z);
            copy(coarse_disp, fine_disp);
            scalar_multiply(fine_disp, fine_mag / magnitude(coarse_disp));
            progs.push_back(DisplacementProgram(coarse_disp, n_coarse, fine_disp, n_fine));
        }

        else{
            cerr << "Expected 4 or 6 strain params but found " << num_read << "\n";
            input.close();
            return false;
        }
    }

    input.close();
    if(progs.size() == 0) cerr << "No strain instructions could be read.\n";
    return progs.size() > 0;
}

//Batch process a set of .nmc files following a specified naming convention
void relaxation_batch(string base_name, string tag, vector<double> mech_params, CGTRParams cgtrpar, vector<DisplacementProgram> progs, bool strain_mode, int n_threads){

    int iter;
    vector<string> net_files, log_files, disp_files, edat_files;

    if(! get_file_lists(base_name, tag, net_files, log_files, disp_files, edat_files)){
        cerr << "File lists could not be obtained.\n";
        return;
    }

    for(iter = 0; iter < net_files.size(); iter ++){
        do_relaxation_run(net_files[iter], log_files[iter], edat_files[iter], disp_files[iter], mech_params, cgtrpar, progs, strain_mode, n_threads);
    }
}

//Ensure a valid input file is supplied. Attempt to process any other command
//line arguments, and upon success, initiate structural relaxation.
int main(int argc, char **argv){

    char c;
    int num_read, n_threads = 1;
    bool strain_mode = true;
    vector<double> mech_params;
    string batch_file = "", base;
    string nmc_base, tag;
    CGTRParams cgtrpar;
    vector<DisplacementProgram> progs;

    //Parse input. Look for a mandatory job file name, and optional parameters
    //controlling the number of threads and determining whether displacements 
    //should be given as fractions of sample height.
    while((c = getopt(argc, argv, "ab:t:")) != -1){
        switch(c) {
            //Use absolute, rather than relative, displacements
            case 'a':
                strain_mode = false;
            case 'b':
                batch_file = string(optarg);
                break;
            case 't':
		        num_read = sscanf(optarg, "%d", &n_threads);
                if(! num_read || n_threads <= 0){
                    cerr << "The number of threads must be a positive integer.\n";
                    return -1;
                }
                break;
            case '?' :
                if(optopt == 'b'){
                    cerr << "Option \"b\" requires a file name.\n";
                    return -1;
                }
                if(optopt == 't'){
                    cerr << "Option \"t\" requires a number of threads.\n";
                    return -1;
                }
                else if(isprint(optopt)){
                    fprintf(stderr, "Unrecognized option: %c\n", optopt);
                }
                else{
                    cerr << stderr, "Unknown option character.\n";
                }
                break;
            default:
                break;
        }
    }

    if(batch_file == ""){
        cerr << "No batch file was supplied.\n";
        exit(1);
    }

    //Attempt to read instructions from the specified job file
    if(! parse_input_file(string(batch_file), base, tag, mech_params, cgtrpar, progs)){
        cerr << "Parsing of the job file failed.\n";
        return -2;
    }

    //Attempt to perform structural relaxation of network-mesh composites
    //specified in files matching a given file pattern, according to the 
    //provided mechanical and relaxation parameters
    relaxation_batch(base, tag, mech_params, cgtrpar, progs, strain_mode, n_threads);

    return 0;
}
