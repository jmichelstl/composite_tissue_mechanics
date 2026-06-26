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
#include "network_gel_relax_utils.hpp"

//Default number of maximum steps for Newton-Raphson root finding
#define DEFAULT_MAX_STEPS 1000

//Size of a trial step when estimating the curvature of the energy along
//a search direction
#define FINITE_DIFF_COEFF 1e-3

#define min(a,b) a <= b ? a : b
#define max(a,b) a >= b ? a : b

//The small dilation to apply to the network to obtain its bulk modulus
#define TRIAL_D 1e-6

using namespace std;

//Struct used in structural relaxation
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

//Report the results of energy relaxation to a file
void report_edata(FILE *edat_file, Vec3D disp, double es, double eb, double eg){

    fprintf(edat_file, "%2.10lf\t%2.10f\t%2.10lf\t%1.10le\t%1.10le\t%1.10le\t%1.10le\n", disp.x, disp.y, disp.z, es, eb, eg, es+eb+eg);
}

//Report data about the quality of convergence, the number of relaxation
//iterations, and the time required
void log_entry(FILE *log_file, double f_rms, int steps, clock_t ct){

    fprintf(log_file, "RMS Force: %1.10le Steps: %d Elapsed time: %lf seconds\n", f_rms, steps, ((double) ct) / CLOCKS_PER_SEC);
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

//Perform a structural relaxation of a composite tissue in a serial manner.
//int nlcg_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, NLCGParams cgpar, double &f_rms){
int nlcg_relax(NetworkMeshData &nmd, vector<double> mech_params, NLCGParams cgpar, double &f_rms){

    int dim, num_dof, offset, cg_iter, cg_max, ls_iter, ls_max, reset_counter;
    vector<double> forces, search_dir, prev_forces;
    double ks, kb, gmod, delta, kt_o_omega, chi, d_curr, d_orig;
    double d_mid, d_prev, ls_coeff, dir_coeff, cg_tol_sqr, sd_mag_sqr;
    double grad_dir_prod, prev_grad_dir_prod;

    //Method to toggle between unified and disjoint methods for computing
    //forces, without having to use large numbers of conditional evaluations
    void(*force_func)(NetworkMeshData &arg1, double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, vector<double> &arg8);

    dim = nmd.points.size();
    offset = 3*nmd.num_net_verts;
    num_dof = dim - offset;

    forces = vector<double>(dim, 0);
    search_dir = vector<double>(num_dof, 0);
    prev_forces = vector<double>(dim, 0);

    //Unpack mechanical parameters
    ks = mech_params[0];
    kb = mech_params[1];
    gmod = mech_params[2];
    delta = mech_params[3];
    kt_o_omega = mech_params[4];
    chi = mech_params[5];

    cg_iter = 0;
    reset_counter = 0;

    //Initialize nonlinear conjugate gradient parameters
    cg_max = cgpar.max_cg_iter;
    ls_max = cgpar.max_ls_iter;
    cg_tol_sqr = cgpar.tol*cgpar.tol;

    //Choose the correct force and energy function pointers, and note the
    //appropriate offset for position updates
    if(nmd.type == nmc_type::unified) {
        force_func = compute_forces_unified;
    }
    else {
        force_func = compute_forces_disjoint;
    }

    //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
    force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces);

    d_curr = cblas_dnrm2(num_dof, &forces[offset], 1);
    d_orig = d_curr;
    cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
    cblas_dcopy(num_dof, &forces[offset], 1, &prev_forces[0], 1);

    do {

        cg_iter ++;

        sd_mag_sqr = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);

        //Compute the inner product of the force a small distance along the
        //search direction with the search direction itself to facilitate the
        //secant line search method.
        cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
        //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
        force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces);
        prev_grad_dir_prod = -cblas_ddot(num_dof,&forces[offset],1,&search_dir[0],1);
        cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
        cblas_dcopy(num_dof, &forces[offset], 1, &prev_forces[0], 1);
        ls_coeff = -FINITE_DIFF_COEFF;

        //Carry out a Newton's Method line search
	    ls_iter = 0;
	    do {

	        //Forces a short distance away along the current search direction
            grad_dir_prod = -cblas_ddot(num_dof,&forces[offset],1,&search_dir[0],1);
            ls_coeff *= grad_dir_prod / (prev_grad_dir_prod - grad_dir_prod);
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &(nmd.points[offset]), 1);

            //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces);
	        ls_iter ++;
            prev_grad_dir_prod = grad_dir_prod;
        } while(ls_iter < ls_max && sd_mag_sqr*ls_coeff*ls_coeff > cg_tol_sqr);

        d_prev = d_curr;
	    d_curr = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);

        reset_counter ++;

        //Periodically reset the search direction, as information about past
        //CG iterations should periodically be discarded.
        if(reset_counter == num_dof){
            reset_counter = 0;
            cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
        }

        else{
            d_mid = cblas_ddot(num_dof, &forces[offset], 1, &prev_forces[0], 1);
            dir_coeff = max((d_curr - d_mid) / d_prev, 0);
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1, &forces[offset], 1, &search_dir[0], 1);
        }

        cblas_dcopy(num_dof, &forces[offset], 1, &prev_forces[0], 1);
    } while(cg_iter < cg_max && d_curr > cg_tol_sqr*d_orig);

    f_rms = cblas_dnrm2(num_dof, &forces[offset], 1) / sqrt(num_dof);
    return cg_iter;
}

//Utility function to compute the total energy, rather than itemized energies,
//using a serial computation for the case that network vertices are also gel
//vertices.
double total_energy_us(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi){

    double energy = 0;

    if(ks > 0 && nmd.edges.size() > 0){
        energy += stretching_energy(nmd.points, nmd.edges, nmd.lengths, ks);
    }

    if(kb > 0 && nmd.triples.size() > 0){
        energy += fiber_bending_energy(nmd.points, nmd.triples, nmd.angles,kb); 
    }

    if(nmd.tetrahedra.size() > 0){
        energy += calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, gmod, delta, kt_o_omega, chi);
    }

    return energy;
}

//Utility function to compute the total energy, rather than itemized energies,
//using a serial computation for the case that network vertices constitute
//a disjoint set relative to the gel vertices.
double total_energy_ds(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi){

    double energy = 0;

    if(ks > 0 && nmd.edges.size() > 0){
        energy += stretching_energy(nmd.points, nmd.edges, nmd.lengths, ks);
    }

    if(kb > 0 && nmd.triples.size() > 0){
        energy += fiber_bending_energy(nmd.points, nmd.triples, nmd.angles,kb); 
    }

    if(nmd.tetrahedra.size() > 0){
        energy += calc_gel_energy_disjoint(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, nmd.reference_points, nmd.vertex_ranges, nmd.contained_vertices, gmod, delta, kt_o_omega, chi);
    }

    return energy;
}

//int cgtr_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, CGTRParams cgtrpar, double &f_rms){
int cgtr_relax(NetworkMeshData &nmd, vector<double> mech_params, CGTRParams cgtrpar, double &f_rms){

    int dim, num_dof, offset, max_outer, max_inner, n_outer, n_inner;
    double max_radius, min_ratio, conv_tol_sqr, radius, cf_model_ratio, curr_cf;
    double prev_cf, curr_mod, prev_mod, sd_hess_norm, ls_coeff, fmag_sqr;
    double fmag_sqr_0, ks, kb, gmod, delta, kt_o_omega, chi, dir_coeff, a, b, c;
    double inner_tol_sqr, rmag_sqr, prev_rmag_sqr, mod_val;
    double prev_mod_val, root_1, root_2, mod_val_1, mod_val_2;
    vector<double> forces, search_dir, hess_dir_proj, step;

    //Means of toggling between the unified and disjoint methods for
    //computing forces
    void(*force_func)(NetworkMeshData &arg1, double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, vector<double> &arg8);
    double(*energy_func)(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double delta_o_omega, double chi);

    dim = nmd.points.size();
    offset = 3*nmd.num_net_verts;
    num_dof = nmd.points.size() - offset;

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

    forces = vector<double>(dim, 0);
    search_dir = vector<double>(num_dof, 0);
    hess_dir_proj = vector<double>(dim, 0);
    step = vector<double>(num_dof, 0);

    if(nmd.type == nmc_type::unified){
        force_func = compute_forces_unified;
        energy_func = total_energy_us;
    }
    else {
        force_func = compute_forces_disjoint;
        energy_func = total_energy_ds;
    }

    //Evaluate the energy and forces at the outset
    //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
    force_func(nmd,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
    fmag_sqr_0 = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);
    fmag_sqr = fmag_sqr_0;
    radius = max_radius;

    prev_cf = energy_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi);

    for(n_outer = 0; n_outer < max_outer; ++n_outer){

        cblas_dscal(num_dof, 0, &step[0], 1);
        rmag_sqr = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);
        prev_rmag_sqr = rmag_sqr;
        cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
        inner_tol_sqr = min(.25, prev_rmag_sqr)*prev_rmag_sqr;
        prev_mod_val = prev_cf;

        //Seek an approximate optimal point for the local model within the
        //current trust region.
        n_inner = 0;
        do if(rmag_sqr > inner_tol_sqr){

            //Find the curvature along the present search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
            //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi,hess_dir_proj);
            cblas_daxpy(num_dof, -1., &forces[offset], 1, &hess_dir_proj[offset], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[offset], 1);
            sd_hess_norm = cblas_ddot(num_dof, &search_dir[0], 1, &hess_dir_proj[offset], 1); 
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);

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
                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[offset], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[offset], 1);
                mod_val_1 += .5*root_1*root_1*sd_hess_norm;

                mod_val_2 = -root_2*cblas_ddot(num_dof, &forces[offset], 1, &search_dir[0], 1);
                mod_val_2 -= 2.*root_2*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[offset], 1);
                mod_val_2 += .5*root_2*root_2*sd_hess_norm;

                if(mod_val_1 < mod_val_2){
                    cblas_daxpy(num_dof, root_1, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_1 + prev_mod_val;
                }
                else{
                    cblas_daxpy(num_dof, root_2, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_2 + prev_mod_val;
                }
                cblas_daxpy(num_dof, 1, &step[0], 1, &(nmd.points[offset]), 1);
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
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &step[0], 1, &(nmd.points[offset]), 1);
            //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi,hess_dir_proj);
            cblas_daxpy(num_dof, -1., &forces[offset], 1, &hess_dir_proj[offset], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[offset], 1);
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &step[0], 1, &(nmd.points[offset]),1);

            prev_rmag_sqr = rmag_sqr;
            rmag_sqr = fmag_sqr + cblas_ddot(num_dof, &hess_dir_proj[offset], 1, &hess_dir_proj[offset], 1) - 2*cblas_ddot(num_dof, &forces[offset], 1, &hess_dir_proj[offset], 1);

            if(rmag_sqr <= inner_tol_sqr) break;

            dir_coeff = rmag_sqr / prev_rmag_sqr;
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1., &forces[offset], 1, &search_dir[0], 1); 
            cblas_daxpy(num_dof, -1., &hess_dir_proj[offset], 1, &search_dir[0], 1); 
            n_inner ++;
        } while(n_inner < max_inner);

        //Evaluate the energy after a step is proposed, and compare the actual
        //and expected reduction. Modify the trust radius as appropriate, and
        //decide whether to keep the search step.
        cblas_daxpy(num_dof, 1., &step[0], 1, &(nmd.points[offset]), 1);

        curr_cf = energy_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi);
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
            //compute_forces(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces);
        }
        else cblas_daxpy(num_dof, -1., &step[0], 1, &(nmd.points[offset]), 1);

        prev_cf = curr_cf;
        fmag_sqr = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);
        if(fmag_sqr / fmag_sqr_0 < conv_tol_sqr) break;
    }

    if(fmag_sqr / fmag_sqr_0 < conv_tol_sqr){
        f_rms = cblas_dnrm2(num_dof, &forces[offset], 1) / sqrt(num_dof);
        return n_outer;
    }

    //Refine the global search with a local nonlinear CG search
    NLCGParams cgpar(max_outer, max_inner, cgtrpar.conv_tol*sqrt(fmag_sqr_0/fmag_sqr));

    //n_outer += nlcg_relax(points, bottom, top, edges, lengths, triples, angles, tets, volumes, inv_mats, mech_params, cgpar, f_rms);
    n_outer += nlcg_relax(nmd, mech_params, cgpar, f_rms);

    return n_outer;
}

//Perform a structural relaxation of a composite tissue in a parallel manner.
//int parallel_nlcg_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, NLCGParams cgpar, int n_threads, double &f_rms){
int parallel_nlcg_relax(NetworkMeshData &nmd, vector<double> mech_params, NLCGParams cgpar, int n_threads, double &f_rms){

    int dim, num_dof, offset, cg_iter, cg_max, ls_iter, ls_max, reset_counter;
    vector<double> forces, search_dir, prev_forces;
    double ks, kb, gmod, delta, kt_o_omega, chi, d_curr, d_orig;
    double d_mid, d_prev, ls_coeff, dir_coeff, cg_tol_sqr, sd_mag_sqr;
    double grad_dir_prod, prev_grad_dir_prod;
    vector<vector<double>> force_sets;
    vector<int> stretch_assignments, bend_assignments, gel_assignments;
    void(*force_func)(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);

    if(nmd.type == nmc_type::unified){
        force_func = compute_forces_unified_parallel;
    }
    else {
        force_func = compute_forces_disjoint_parallel;
    }

    dim = nmd.points.size();
    offset = 3*nmd.num_net_verts;
    num_dof = nmd.points.size() - offset;
    cg_iter = reset_counter = 0;

    //Allocate sets of auxilliary force vectors to enable parallel computation
    for(int iter = 0; iter < n_threads; iter++){
        force_sets.push_back(vector<double>(dim, 0));
    }

    //Determine how jobs should be allotted to threads for each type of force
    //to be computed.
    stretch_assignments = job_assignments(nmd.edges.size()/2, n_threads);
    bend_assignments = job_assignments(nmd.triples.size()/3, n_threads);
    gel_assignments = job_assignments(nmd.tetrahedra.size()/4, n_threads);

    forces = vector<double>(dim, 0);
    search_dir = vector<double>(num_dof, 0);
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

    //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
    force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);

    d_curr = cblas_dnrm2(num_dof, &forces[offset], 1);
    d_orig = d_curr;
    cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
    cblas_dcopy(num_dof, &forces[offset], 1, &prev_forces[0], 1);

    do {

        cg_iter ++;

        sd_mag_sqr = cblas_ddot(num_dof, &search_dir[0], 1, &search_dir[0], 1);

        //Compute the inner product of the force with the search direction
        //to facilitate the secant method
        cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
        //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
        force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
        prev_grad_dir_prod = -cblas_ddot(num_dof, &forces[offset], 1, &search_dir[0],1);
        cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
        cblas_dcopy(num_dof, &prev_forces[0], 1, &forces[offset], 1);
        ls_coeff = -FINITE_DIFF_COEFF;


        //Carry out a Newton's Method line search
        ls_iter = 0;
        do {

            //Forces a short distance away along the current search direction
            grad_dir_prod = -cblas_ddot(num_dof, &forces[offset], 1, &search_dir[0], 1);
            ls_coeff *= grad_dir_prod / (prev_grad_dir_prod - grad_dir_prod);
            cblas_daxpy(num_dof, ls_coeff, &search_dir[0], 1, &(nmd.points[offset]), 1);
            prev_grad_dir_prod = grad_dir_prod;
            ls_iter ++;

            //Forces at the current position
            //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);

        } while(ls_iter < ls_max && sd_mag_sqr*ls_coeff*ls_coeff > cg_tol_sqr);

        d_prev = d_curr;
        d_curr = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);

        reset_counter ++;
        //Periodically reset the search direction, as information about past
        //CG iterations should periodically be discarded.
        if(reset_counter == num_dof){
            reset_counter = 0;
	        cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
        }

        else{
            d_mid = cblas_ddot(num_dof, &forces[offset], 1, &prev_forces[0], 1);
            dir_coeff = max((d_curr - d_mid) / d_prev, 0);
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1, &forces[offset], 1, &search_dir[0], 1);
        }

        cblas_dcopy(num_dof, &forces[offset], 1, &prev_forces[0], 1);
    } while(cg_iter < cg_max && d_curr > cg_tol_sqr*d_orig);

    f_rms = cblas_dnrm2(num_dof, &forces[offset], 1) / sqrt(num_dof);
    return cg_iter;
}

//Compute the total energy using parallelized routines in the case that the
//network vertices are a subset of the gel vertices
double total_energy_up(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi){

    double energy = 0;

    if(nmd.edges.size() > 0){
        energy += parallel_stretching_energy(nmd.points, nmd.edges, nmd.lengths, ks);
    }

    if(nmd.triples.size() > 0){
        energy += parallel_fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, kb);
    }

    energy += parallel_calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, gmod, delta, kt_o_omega, chi);

    return energy;
}

//Compute the total energy using parallelized routines in the case that the
//network vertices are a disjoint set from the gel vertices
double total_energy_dp(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi){

    double energy = 0;

    if(nmd.edges.size() > 0){
        energy += parallel_stretching_energy(nmd.points, nmd.edges, nmd.lengths, ks);
    }

    if(nmd.triples.size() > 0){
        energy += parallel_fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, kb);
    }

    energy += parallel_calc_gel_energy_disjoint(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, nmd.reference_points, nmd.vertex_ranges, nmd.contained_vertices, gmod, delta, kt_o_omega, chi);

    return energy;
}

//int parallel_cgtr_relax(vector<double> &points, unordered_set<int> bottom, unordered_set<int> top, vector<int> edges, vector<double> lengths, vector<int> triples, vector<double> angles, vector<int> tets, vector<double> volumes, vector<double> inv_mats, vector<double> mech_params, CGTRParams cgtrpar, int n_threads, double &f_rms){
int parallel_cgtr_relax(NetworkMeshData &nmd, vector<double> mech_params, CGTRParams cgtrpar, int n_threads, double &f_rms){

    int dim, num_dof, offset, max_outer, max_inner, n_outer, n_inner;
    double max_radius, min_ratio, conv_tol_sq, radius, cf_model_ratio, curr_cf;
    double prev_cf, curr_mod, prev_mod, sd_hess_norm, ls_coeff, fmag_sqr;
    double fmag_sqr_0, ks, kb, gmod, delta, kt_o_omega, chi, dir_coeff, a, b, c;
    double inner_tol_sqr, rmag_sqr, prev_rmag_sqr, mod_val, prev_mod_val;
    double root_1, root_2, mod_val_1, mod_val_2;
    vector<double> forces, search_dir, hess_dir_proj, step;
    vector<vector<double>> force_sets;
    vector<int> stretch_assignments, bend_assignments, gel_assignments;

    //Means of toggling between unified and disjoint approaches to force
    //and energy computations
    void(*force_func)(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments);
    double(*energy_func)(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi);

    dim = nmd.points.size();
    offset = nmd.num_net_verts * 3;
    num_dof = dim - offset;

    //Choose the correct procedures for force and energy computation
    if(nmd.type == nmc_type::unified){
        force_func = compute_forces_unified_parallel;
        energy_func = total_energy_up;
    }
    else {
        force_func = compute_forces_disjoint_parallel;
        energy_func = total_energy_dp;
    }

    //Allocate sets of auxilliary force vectors to enable parallel computation
    for(int iter = 0; iter < n_threads; iter++){
        force_sets.push_back(vector<double>(num_dof, 0));
    }
    //Determine how jobs should be allotted to threads for each type of force
    //to be computed.
    stretch_assignments = job_assignments(nmd.edges.size()/2, n_threads);
    bend_assignments = job_assignments(nmd.triples.size()/3, n_threads);
    gel_assignments = job_assignments(nmd.tetrahedra.size()/4, n_threads);

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

    forces = vector<double>(dim, 0);
    search_dir = vector<double>(num_dof, 0);
    hess_dir_proj = vector<double>(dim, 0);
    step = vector<double>(num_dof, 0);

    //Evaluate the energy and forces at the outset
    //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
    force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
    fmag_sqr_0 = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);
    fmag_sqr = fmag_sqr_0;
    radius = max_radius;

    prev_cf = energy_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi);

    for(n_outer = 0; n_outer < max_outer; ++n_outer){

        cblas_dscal(num_dof, 0, &step[0], 1);
        rmag_sqr = cblas_ddot(num_dof, &forces[offset], 1, &forces[offset], 1);
        prev_rmag_sqr = rmag_sqr;
        cblas_dcopy(num_dof, &forces[offset], 1, &search_dir[0], 1);
        inner_tol_sqr = min(.25, prev_rmag_sqr)*prev_rmag_sqr;
        prev_mod_val = prev_cf;

        //Seek an approximate optimal point for the local model within the
        //current trust region.
        for(n_inner = 0; n_inner < max_inner; ++n_inner){

            //Find the curvature along the present search direction
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);
            //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            cblas_daxpy(num_dof, -1., &forces[offset], 1, &hess_dir_proj[offset], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[offset], 1);
            sd_hess_norm = cblas_ddot(num_dof, &search_dir[0], 1, &hess_dir_proj[offset], 1); 
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &search_dir[0], 1, &(nmd.points[offset]), 1);

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
                mod_val_1 = -root_1*cblas_ddot(num_dof, &forces[offset], 1, &search_dir[0], 1);
                mod_val_1 -= 2.*root_1*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[offset], 1);
                mod_val_1 += .5*root_1*root_1*sd_hess_norm;

                mod_val_2 = -root_2*cblas_ddot(num_dof, &forces[0], 1, &search_dir[0], 1);
                mod_val_2 -= 2.*root_2*cblas_ddot(num_dof, &step[0], 1, &hess_dir_proj[0], 1);
                mod_val_2 += .5*root_2*root_2*sd_hess_norm;

                if(mod_val_1 < mod_val_2){
                    cblas_daxpy(num_dof, root_1, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_1 + prev_mod_val;
                }
                else{
                    cblas_daxpy(num_dof, root_2, &search_dir[0], 1, &step[0], 1);
                    mod_val = mod_val_2 + prev_mod_val;
                }
                cblas_daxpy(num_dof, 1, &step[0], 1, &(nmd.points[offset]), 1);
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
            cblas_daxpy(num_dof, FINITE_DIFF_COEFF, &step[0], 1, &(nmd.points[offset]), 1);
            //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, hess_dir_proj, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            cblas_daxpy(num_dof, -1., &forces[offset], 1, &hess_dir_proj[offset], 1);
            cblas_dscal(num_dof, -1./FINITE_DIFF_COEFF, &hess_dir_proj[offset], 1);
            cblas_daxpy(num_dof, -FINITE_DIFF_COEFF, &step[0], 1, &(nmd.points[offset]),1);

            prev_rmag_sqr = rmag_sqr;
            rmag_sqr = fmag_sqr + cblas_ddot(num_dof, &hess_dir_proj[offset], 1, &hess_dir_proj[offset], 1) - 2*cblas_ddot(num_dof, &forces[offset], 1, &hess_dir_proj[offset], 1);

            if(rmag_sqr <= inner_tol_sqr) break;

            dir_coeff = rmag_sqr / prev_rmag_sqr;
            cblas_dscal(num_dof, dir_coeff, &search_dir[0], 1);
            cblas_daxpy(num_dof, 1., &forces[offset], 1, &search_dir[0], 1); 
            cblas_daxpy(num_dof, -1., &hess_dir_proj[offset], 1, &search_dir[0], 1); 
        }

        //Evaluate the energy after a step is proposed, and compare the actual
        //and expected reduction. Modify the trust radius as appropriate, and
        //decide whether to keep the search step.
        cblas_daxpy(num_dof, 1., &step[0], 1, &(nmd.points[offset]), 1);
        curr_cf = energy_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi);

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
            //compute_forces_parallel(points, top, bottom, edges, lengths, triples, angles, tets, volumes, inv_mats,  ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
            force_func(nmd, ks, kb, gmod, delta, kt_o_omega, chi, forces, force_sets, stretch_assignments, bend_assignments, gel_assignments);
        }
        else cblas_daxpy(num_dof, -1., &step[0], 1, &(nmd.points[offset]), 1);

        prev_cf = curr_cf;
    }

    if(fmag_sqr / fmag_sqr_0 < conv_tol_sq){
        f_rms = cblas_dnrm2(num_dof, &forces[offset], 1) / sqrt(num_dof);
        return n_outer;
    }

    //Refine the global search with a local nonlinear CG search
    NLCGParams cgpar(max_outer, max_inner, cgtrpar.conv_tol*sqrt(fmag_sqr_0/fmag_sqr));

    //n_outer += parallel_nlcg_relax(points, bottom, top, edges, lengths, triples, angles, tets, volumes, inv_mats, mech_params, cgpar, n_threads, f_rms);
    n_outer += parallel_nlcg_relax(nmd, mech_params, cgpar, n_threads, f_rms);

    f_rms = cblas_dnrm2(num_dof, &forces[offset], 1) / sqrt(num_dof);
    return n_outer;
}

//Direct a course of deformation and structural relaxation. Given the specified
//number of threads, use either the serial or parallel structural relaxation
//function. Report the relaxed energy, deformed coordinates, and logfile
//information for each coarse increment.
void do_relaxation_run(string nmc_file, string log_name, string edat_name, string nmt_base, vector<double> mech_params, CGTRParams par, vector<DisplacementProgram> progs, bool strain_mode, int n_threads, nmc_type type){

    int n_steps, num_digits, i, j;
    NetworkMeshData nmd(type);
    double dilation, volume, estretch, ebend, egel, k_network, f_rms, swelling;
    /*vector<double> bounds, points, lengths, angles, volumes, inv_mats;
    unordered_set<int> top, bottom;
    vector<int> edges, triples, elements;*/
    Vec3D total_displacement, displacement, coarse_step, fine_step;
    FILE *edat_file, *log_file;
    string xml_name, nmt_name;
    //Data type for timing the relaxation run
    clock_t ct;
    bool parallel = n_threads > 1;

    //double minx, maxx, miny, maxy, minz, maxz;

    //Attempt to read a network mesh composite (NMC) file, and intialize
    //vertices, edges, triples, mesh elements, and associated data structures.
    //if(! parse_nmc_file(nmc_file, bounds, points, bottom, top, edges, lengths, triples, angles, elements)){
    if(! parse_nmc_file(nmc_file, nmd)){
        cerr << "Parsing of the network mesh complex file failed.\n";
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

    volume = (nmd.bounds[1]-nmd.bounds[0])*(nmd.bounds[3]-nmd.bounds[2])*(nmd.bounds[5]-nmd.bounds[4]);

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
    dilate_vertices(nmd.points, nmd.bounds, 1/swelling);

    //Find tetrahedron volumes
    calc_volumes(nmd);

    //Make a record of the reference positions of network vertices
    nmd.reference_points.assign(nmd.points.begin(), nmd.points.begin()+3*nmd.num_net_verts);

    //Find matrices for computing local affine transformations
    get_inv_mats(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.type);

    //If the network-mesh complex is disjoint, initialize a SuiteSparse CHOLMOD
    //context and find the sparse matrix mapping network forces onto gel forces
    if(type == nmc_type::disjoint){
        cholmod_l_start(&(nmd.common));
        compute_network_mesh_map(nmd);
    }

    dilate_vertices(nmd.points, nmd.bounds, swelling);
    total_displacement = Vec3D(0, 0, 0);
    estretch = stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
    cout << "Stretching energy after dilation: " << estretch << endl;
    egel = calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    cout << "Gel energy: " << egel << endl;

    /*minx = FLT_MAX;
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
    */

    ct = clock();
    if(! parallel){
        //n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
        n_steps = cgtr_relax(nmd, mech_params, par, f_rms);
    }
    else {
        //n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
        n_steps = parallel_cgtr_relax(nmd, mech_params, par, n_threads, f_rms);
    }
    ct = clock() - ct;

    if(! parallel){
        estretch = stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
	    ebend = fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
        egel = calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    }
    else{
        estretch = parallel_stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
	    ebend = parallel_fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
	    egel = parallel_calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
    }
    report_edata(edat_file, total_displacement, estretch, ebend, egel);
    log_entry(log_file, f_rms, n_steps, ct);

    //If position data are to be recorded, record the initial, swollen state of
    //the network.
    if(nmt_base != ""){
        if(!append_network_mesh_state(total_displacement, nmd.points, xml_name, nmt_name)){
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
            scalar_multiply(coarse_step, nmd.bounds[5] - nmd.bounds[4]);
        }

        //Find the fine increment, if a fine increment is to be used
        if(next_prog.num_fine > 0){
            fine_step = next_prog.fine_step;
            if(strain_mode){
                scalar_multiply(fine_step, nmd.bounds[5] - nmd.bounds[4]);
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
        /*egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
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
        cout << "Bounds before: (" << minx << ", " << maxx << "), (" << miny << ", " << maxy << "), (" << minz << ", " << maxz << ")\n";*/
        affine_deform(nmd.points, nmd.bounds, displacement);
        /*egel = calc_gel_energy(points, elements, inv_mats, volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
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
        cout << "Bounds after: (" << minx << ", " << maxx << "), (" << miny << ", " << maxy << "), (" << minz << ", " << maxz << ")\n";*/
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
                        //n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
                        n_steps = cgtr_relax(nmd, mech_params, par, f_rms);
                    }
                    else {
                        //n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
                        n_steps = parallel_cgtr_relax(nmd, mech_params, par, n_threads, f_rms);
                    }
                    ct = clock() - ct;
                    if(! parallel){
                        estretch = stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
                        ebend = fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
                        egel = calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                    }
                    else{
                        estretch = parallel_stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
                        ebend = parallel_fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
                        egel = parallel_calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                    }
                    report_edata(edat_file, total_displacement, estretch, ebend, egel);
                    log_entry(log_file, f_rms, n_steps, ct);

                    //Only report displacements at coarse intervals
                    if(j == next_prog.num_fine){
                        if(nmt_base != ""){
                            if(!append_network_mesh_state(total_displacement, nmd.points, xml_name, nmt_name)){
                                cerr << "Appending of position data failed.\n";
                                fclose(log_file);
                                fclose(edat_file);
                                return;
                            }
                        }

                    }
                    if(j < 2*next_prog.num_fine){
                        affine_deform(nmd.points, nmd.bounds, fine_step);
			            axpy(fine_step, total_displacement, 1.);
                    }
                }

            }

            else{
                ct = clock();
                if(! parallel){
                    //n_steps = cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, f_rms);
                    n_steps = cgtr_relax(nmd, mech_params, par, f_rms);
                }
                else {
                    //n_steps = parallel_cgtr_relax(points, bottom, top, edges, lengths, triples, angles, elements, volumes, inv_mats, mech_params, par, n_threads, f_rms);
                    n_steps = parallel_cgtr_relax(nmd, mech_params, par, n_threads, f_rms);
                }
                ct = clock() - ct;

                if(! parallel){
                    estretch = stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
                    ebend = fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
                    egel = calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                }
                else{
                    estretch = parallel_stretching_energy(nmd.points, nmd.edges, nmd.lengths, mech_params[0]);
                    ebend = parallel_fiber_bending_energy(nmd.points, nmd.triples, nmd.angles, mech_params[1]);
                    egel = parallel_calc_gel_energy(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, mech_params[2], mech_params[3], mech_params[4], mech_params[5]);
                }
                report_edata(edat_file, total_displacement, estretch, ebend, egel);
                log_entry(log_file, f_rms, n_steps, ct);

    	        if(nmt_base != ""){
                    if(!append_network_mesh_state(total_displacement, nmd.points, xml_name, nmt_name)){
                        cerr << "Appending of position data failed.\n";
                        fclose(log_file);
                        fclose(edat_file);
                        return;
                    }
                }
            }

            if(i < next_prog.num_coarse - 1){
                affine_deform(nmd.points, nmd.bounds, displacement);
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

    //If the network-mesh complex is disjoint, free sparse matrix resources
    if(type == nmc_type::disjoint){
        cholmod_l_free_sparse(&nmd.force_mapping, &(nmd.common));
        cholmod_l_finish(&(nmd.common));
    }
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
void relaxation_batch(string base_name, string tag, string extension, vector<double> mech_params, CGTRParams cgtrpar, vector<DisplacementProgram> progs, bool strain_mode, int n_threads, nmc_type type){

    int iter;
    vector<string> net_files, log_files, disp_files, edat_files;

    if(! get_file_lists(base_name, tag, extension, net_files, log_files, disp_files, edat_files)){
        cerr << "File lists could not be obtained.\n";
        return;
    }

    for(iter = 0; iter < net_files.size(); iter ++){
        do_relaxation_run(net_files[iter], log_files[iter], edat_files[iter], disp_files[iter], mech_params, cgtrpar, progs, strain_mode, n_threads, type);
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
    string nmc_base, tag, extension;
    CGTRParams cgtrpar;
    vector<DisplacementProgram> progs;
    nmc_type type = nmc_type::unified;

    //By default, look for .nmc extension
    extension = "nmc";

    //Parse input. Look for a mandatory job file name, and optional parameters
    //controlling the number of threads and determining whether displacements 
    //should be given as fractions of sample height.
    while((c = getopt(argc, argv, "ab:dt:")) != -1){
        switch(c) {
            //Use absolute, rather than relative, displacements
            case 'a':
                strain_mode = false;
            case 'b':
                batch_file = string(optarg);
                break;
            case 'd':
                type = nmc_type::disjoint;
                extension = "dnmc";
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
    relaxation_batch(base, tag, extension, mech_params, cgtrpar, progs, strain_mode, n_threads, type);

    return 0;
}
