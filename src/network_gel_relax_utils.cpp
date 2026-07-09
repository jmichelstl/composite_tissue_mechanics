#include "network_gel_relax_utils.hpp"
#include <cfloat>
#include <iostream>

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
void calc_volumes(NetworkMeshData &nmd){

    Vec3D p1, p2, p3, p4, d1, d2, d3, cross_prod;

    for(int iter = 0; iter < nmd.tetrahedra.size() / 4; iter ++){
        load_coords(nmd.points, nmd.tetrahedra[iter*4], p1);
        load_coords(nmd.points, nmd.tetrahedra[iter*4+1], p2);
        load_coords(nmd.points, nmd.tetrahedra[iter*4+2], p3);
        load_coords(nmd.points, nmd.tetrahedra[iter*4+3], p4);

        difference(p2, p1, d1);
        difference(p3, p1, d2);
        difference(p4, p1, d3);

        cross(d2, d3, cross_prod);

        nmd.volumes.push_back(abs(dot(d1, cross_prod)) / 6);
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
void get_inv_mats(vector<double> points, vector<int> tets, vector<double> &inv_mats, nmc_type type){

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

        //Row 1
        inv_mats.push_back((y2*(z3 - z4) + y3*(z4 - z2) + y4*(z2 - z3)) * inv_det);
        inv_mats.push_back((x2*(z4 - z3) + x3*(z2 - z4) + x4*(z3 - z2)) * inv_det);
        inv_mats.push_back((x4*(y2 - y3) + x2*(y3 - y4) + x3*(y4 - y2)) * inv_det);
        if(type == nmc_type::disjoint){
            inv_mats.push_back((x2*(y4*z3-y3*z4) + x3*(y2*z4-y4*z2) + x4*(y3*z2-y2*z3))*inv_det);
        }

        //Row 2
        inv_mats.push_back((y1*(z4 - z3) + y3*(z1 - z4) + y4*(z3 - z1)) * inv_det);
        inv_mats.push_back((x1*(z3 - z4) + x3*(z4 - z1) + x4*(z1 - z3)) * inv_det);
        inv_mats.push_back((x1*(y4 - y3) + x3*(y1 - y4) + x4*(y3 - y1)) * inv_det);
        if(type == nmc_type::disjoint){
            inv_mats.push_back((x1*(y3*z4-y4*z3)+x3*(y4*z1-y1*z4)+x4*(y1*z3-y3*z1))*inv_det);
        }

        //Row 3
        inv_mats.push_back((y1*(z2 - z4) + y2*(z4 - z1) + y4*(z1 - z2)) * inv_det);
        inv_mats.push_back((x1*(z4 - z2) + x2*(z1 - z4) + x4*(z2 - z1)) * inv_det);
        inv_mats.push_back((x1*(y2 - y4) + x2*(y4 - y1) + x4*(y1 - y2)) * inv_det);
        if(type == nmc_type::disjoint){
            inv_mats.push_back((x1*(y4*z2-y2*z4)+x2*(y1*z4-y4*z1)+x4*(y2*z1-y1*z2))*inv_det);
        }

        //Row 4
        inv_mats.push_back((y1*(z3 - z2) + y2*(z1 - z3) + y3*(z2 - z1)) * inv_det);
        inv_mats.push_back((x1*(z2 - z3) + x2*(z3 - z1) + x3*(z1 - z2)) * inv_det);
        inv_mats.push_back((x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1)) * inv_det);
        if(type == nmc_type::disjoint){
            inv_mats.push_back((x1*(y2*z3-y3*z2)+x2*(y3*z1-y1*z3)+x3*(y1*z2-y2*z1))*inv_det);
        }
    }
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

//Compute stretching energy in a parallel manner
double parallel_stretching_energy(vector<double> points, vector<int> edges, vector<double> lengths, double ks){

    vector<double> energies(edges.size(), 0);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, edges.size()/3),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                stretching_energy_worker(points, edges, lengths, ks, index, &energies[index]);
            }
        }
    );

    return sum_reduce(&energies[0], &energies[0]+energies.size(), 0.);
}

//Compute fiber bending forces in a serial manner.
void bending_forces(vector<double> pts, vector<int> triples, vector<double> angles, double kb, vector<double> &forces, int start, int end){

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
        det = f11*(f22*f33-f23*f32) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);
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

void gel_forces_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi, vector<double> &forces, int start, int end){

    //Components of the affine transformation matrix, derivatives of the
    //gel elastic energy with respect to the deformation gradient tensor
    //components, and an augmented coordinate of a network vertex
    double de_df11, de_df12, de_df13, de_df21, de_df22, de_df23, de_df31;
    double de_df32, de_df33, ref_coord[4], disp_coord[4], x[16], a[16];

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and coefficients multiplying deformation gradient tensor
    //components to calculate forces
    double det, sq_sum, coeff1, coeff2;
    int idx1, idx2, idx3, idx4, i;

    vector<double> inv_mat;

    ref_coord[3] = 1.;
    for(i = 12; i < 16; i++) x[i] = 1.;

    for(int tet_idx = start; tet_idx != end; tet_idx ++){

        //Integer indices of the vertices of the tetrahedron
        idx1 = tetrahedra[tet_idx*4];
        idx2 = tetrahedra[tet_idx*4+1];
        idx3 = tetrahedra[tet_idx*4+2];
        idx4 = tetrahedra[tet_idx*4+3];

        //Compute the matrix for computing an affine transformation for the
        //current tetrahedral cell
	    inv_mat.assign(inv_mats.begin()+16*tet_idx, inv_mats.begin()+16*(tet_idx+1));

        for(i = 0; i < 3; i++){
            x[i*4] = pts[idx1*3+i];
            x[i*4+1] = pts[idx2*3+i];
            x[i*4+2] = pts[idx3*3+i];
            x[i*4+3] = pts[idx4*3+i];
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1., x, 4, &inv_mat[0], 4, 0, a, 4);
        
        for(i = vertex_ranges[tet_idx]; i < vertex_ranges[tet_idx+1]; i++){
            cblas_dcopy(3, &ref_pts[3*contained_vertices[i]], 1, ref_coord, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1., a, 4, ref_coord, 1, 0, disp_coord, 1);
            cblas_dcopy(3, disp_coord, 1, &pts[3*contained_vertices[i]], 1);
        }

        //Find derivatives of the gel elastic energy with respect to the 
	    //components of the deformation gradient tensor.
        det = a[0]*(a[5]*a[10]-a[6]*a[9]) + a[1]*(a[6]*a[8]-a[4]*a[10]) + a[2]*(a[4]*a[9]-a[5]*a[8]);
        sq_sum = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[4]*a[4] + a[5]*a[5];
        sq_sum += a[6]*a[6] + a[8]*a[8] + a[9]*a[9] + a[10]*a[10];
        coeff1 = -g*volumes[tet_idx];
        coeff2 = volumes[tet_idx] * (g*(1 + excess_osm) / det - kt_o_omega * (log((det - 1) / det) + (det+chi)/(det*det)));

        de_df11 = coeff1*a[0] + coeff2*(a[5]*a[10] - a[6]*a[9]);
        de_df12 = coeff1*a[1] + coeff2*(a[6]*a[8] - a[4]*a[10]);
        de_df13 = coeff1*a[2] + coeff2*(a[4]*a[9] - a[5]*a[8]);
        de_df21 = coeff1*a[4] + coeff2*(a[2]*a[9] - a[1]*a[19]);
        de_df22 = coeff1*a[5] + coeff2*(a[0]*a[10] - a[2]*a[8]);
        de_df23 = coeff1*a[6] + coeff2*(a[1]*a[8] - a[0]*a[9]);
        de_df31 = coeff1*a[8] + coeff2*(a[1]*a[6] - a[2]*a[5]);
        de_df32 = coeff1*a[9] + coeff2*(a[2]*a[4] - a[0]*a[6]);
        de_df33 = coeff1*a[10] + coeff2*(a[0]*a[5] - a[1]*a[4]);

        //Find forces on each vertex using the chain rule
        forces[idx1*3] += de_df11*inv_mat[0] + de_df12*inv_mat[1] + de_df13*inv_mat[2];
        forces[idx1*3+1] += de_df21*inv_mat[0] + de_df22*inv_mat[1] + de_df23*inv_mat[2];
        forces[idx1*3+2] += de_df31*inv_mat[0] + de_df32*inv_mat[1] + de_df33*inv_mat[2];

        forces[idx2*3] += de_df11*inv_mat[4] + de_df12*inv_mat[5] + de_df13*inv_mat[6];
        forces[idx2*3+1] += de_df21*inv_mat[4] + de_df22*inv_mat[5] + de_df23*inv_mat[6];
        forces[idx2*3+2] += de_df31*inv_mat[4] + de_df32*inv_mat[5] + de_df33*inv_mat[6];

        forces[idx3*3] += de_df11*inv_mat[8] + de_df12*inv_mat[9] + de_df13*inv_mat[10];
        forces[idx3*3+1] += de_df21*inv_mat[8] + de_df22*inv_mat[9] + de_df23*inv_mat[10];
        forces[idx3*3+2] += de_df31*inv_mat[8] + de_df32*inv_mat[9] + de_df33*inv_mat[10];

        forces[idx4*3] += de_df11*inv_mat[12] + de_df12*inv_mat[13] + de_df13*inv_mat[14];
        forces[idx4*3+1] += de_df21*inv_mat[12] + de_df22*inv_mat[13] + de_df23*inv_mat[14];
        forces[idx4*3+2] += de_df31*inv_mat[12] + de_df32*inv_mat[13] + de_df33*inv_mat[14];
    }
}

//Find the deformation energies of gel elements in a serial manner.
double calc_gel_energy(vector<double> pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, double g, double excess_osm, double kt_o_omega, double chi){

    //Components of the deformation gradient tensor
    double f11, f12, f13, f21, f22, f23, f31, f32, f33;

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
        det = f11*(f22*f33-f23*f32) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);

        sq_sum = f11*f11 + f12*f12 + f13*f13 + f21*f21 + f22*f22 + f23*f23;
        sq_sum += f31*f31 + f32*f32 + f33*f33;
        energy += volumes[tet_idx] * g * .5 * (sq_sum - 3 - 2*(1 + excess_osm)*log(det));
        energy += volumes[tet_idx] * kt_o_omega * ((det - 1)*log((det - 1) / det) + chi*(det - 1)/det);
    }

    return energy;
}

double calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi){

    //Components of the affine transformation matrix and an augmented 
    //coordinate of a network vertex
    double de_df11, de_df12, de_df13, de_df21, de_df22, de_df23, de_df31;
    double de_df32, de_df33, ref_coord[4], disp_coord[4], x[16], a[16];

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and energy
    double det, sq_sum, energy;
    int idx1, idx2, idx3, idx4, i;

    vector<double> inv_mat;

    energy = 0;

    ref_coord[3] = 1.;
    for(i = 12; i < 16; i++) x[i] = 1.;

    for(int tet_idx = 0; tet_idx < tetrahedra.size()/4; tet_idx ++){

        //Integer indices of the vertices of the tetrahedron
        idx1 = tetrahedra[tet_idx*4];
        idx2 = tetrahedra[tet_idx*4+1];
        idx3 = tetrahedra[tet_idx*4+2];
        idx4 = tetrahedra[tet_idx*4+3];

        //Compute the matrix for computing an affine transformation for the
        //current tetrahedral cell
	    inv_mat.assign(inv_mats.begin()+16*tet_idx, inv_mats.begin()+16*(tet_idx+1));

        for(i = 0; i < 3; i++){
            x[i*4] = pts[idx1*3+i];
            x[i*4+1] = pts[idx2*3+i];
            x[i*4+2] = pts[idx3*3+i];
            x[i*4+3] = pts[idx4*3+i];
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1., x, 4, &inv_mat[0], 4, 0, a, 4);
        
        for(i = vertex_ranges[tet_idx]; i < vertex_ranges[tet_idx+1]; i++){
            cblas_dcopy(3, &ref_pts[3*contained_vertices[i]], 1, ref_coord, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1., a, 4, ref_coord, 1, 0, disp_coord, 1);
            cblas_dcopy(3, disp_coord, 1, &pts[3*contained_vertices[i]], 1);
        }

        //Find the gel elastic energy of the present mesh cell
        det = a[0]*(a[5]*a[10]-a[6]*a[9]) + a[1]*(a[6]*a[8]-a[4]*a[10]) + a[2]*(a[4]*a[9]-a[5]*a[8]);
        sq_sum = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[4]*a[4] + a[5]*a[5];
        sq_sum += a[6]*a[6] + a[8]*a[8] + a[9]*a[9] + a[10]*a[10];

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
    det = f11*(f22*f33-f23*f32) + f12*(f23*f31-f21*f33) +f13*(f21*f32-f22*f31);
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

void disjoint_gel_energy_worker(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi, int tet_index, double *energy){

    //Components of the affine transformation matrix and an augmented 
    //coordinate of a network vertex
    double de_df11, de_df12, de_df13, de_df21, de_df22, de_df23, de_df31;
    double de_df32, de_df33, ref_coord[4], disp_coord[4], x[16], a[16];

    //Determinant, trace of the deformation gradient tensor multiplied by its
    //transpose, and energy
    double det, sq_sum;
    int idx1, idx2, idx3, idx4, i;

    vector<double> inv_mat;

    energy = 0;

    ref_coord[3] = 1.;
    for(i = 12; i < 16; i++) x[i] = 1.;

    //Integer indices of the vertices of the tetrahedron
    idx1 = tetrahedra[tet_index*4];
    idx2 = tetrahedra[tet_index*4+1];
    idx3 = tetrahedra[tet_index*4+2];
    idx4 = tetrahedra[tet_index*4+3];

    //Compute the matrix for computing an affine transformation for the
    //current tetrahedral cell
    inv_mat.assign(inv_mats.begin()+16*tet_index, inv_mats.begin()+16*(tet_index+1));

    for(i = 0; i < 3; i++){
        x[i*4] = pts[idx1*3+i];
        x[i*4+1] = pts[idx2*3+i];
        x[i*4+2] = pts[idx3*3+i];
        x[i*4+3] = pts[idx4*3+i];
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1., x, 4, &inv_mat[0], 4, 0, a, 4);
        
    for(i = vertex_ranges[tet_index]; i < vertex_ranges[tet_index+1]; i++){
        cblas_dcopy(3, &ref_pts[3*contained_vertices[i]], 1, ref_coord, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1., a, 4, ref_coord, 1, 0, disp_coord, 1);
        cblas_dcopy(3, disp_coord, 1, &pts[3*contained_vertices[i]], 1);
    }

    //Find the gel elastic energy of the present mesh cell
    det = a[0]*(a[5]*a[10]-a[6]*a[9]) + a[1]*(a[6]*a[8]-a[4]*a[10]) + a[2]*(a[4]*a[9]-a[5]*a[8]);
    sq_sum = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[4]*a[4] + a[5]*a[5];
    sq_sum += a[6]*a[6] + a[8]*a[8] + a[9]*a[9] + a[10]*a[10];

    *energy = volumes[tet_index] * g * .5 * (sq_sum - 3 - 2*(1 + excess_osm)*log(det));
    *energy += volumes[tet_index] * kt_o_omega * ((det - 1)*log((det - 1) / det) + chi*(det - 1)/det);
}

double parallel_calc_gel_energy_disjoint(vector<double> &pts, vector<int> tetrahedra, vector<double> inv_mats, vector<double> volumes, vector<double> ref_pts, vector<int> vertex_ranges, vector<int> contained_vertices, double g, double excess_osm, double kt_o_omega, double chi){

    vector<double> energies(tetrahedra.size() / 4, 0);

    //Compute energies in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tetrahedra.size()/4),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                disjoint_gel_energy_worker(pts, tetrahedra, inv_mats, volumes, ref_pts, vertex_ranges, contained_vertices, g, excess_osm, kt_o_omega, chi, index, &energies[index]);
            }
        }
    );

    //Sum energies using a parallel reduction
    return sum_reduce(&energies[0], &energies[0]+energies.size(), 0.);
}

//Helper function for computation of restoring forces when the network vertices
//are a subset of the mesh vertices
void compute_forces_unified(NetworkMeshData &nmd,  double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces){

    cblas_dscal(forces.size(), 0, &forces[0], 1);

    gel_forces(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, gmod, delta, kt_o_omega, chi, forces, 0, nmd.tetrahedra.size() / 4);
    //cout << "Gel force norm: " << cblas_dnrm2(nmd.points.size(), &forces[0], 1) << endl;

    if(ks > 0){
        stretching_forces(nmd.points, nmd.edges, nmd.lengths, ks, forces, 0, nmd.edges.size()/2);
    }

    //cout << "Stretching force norm: " << cblas_dnrm2(nmd.points.size(), &forces[0], 1) << endl;

    if(kb > 0){
        bending_forces(nmd.points, nmd.triples, nmd.angles, kb, forces, 0, nmd.triples.size()/3);
    }

    //gel_forces(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, gmod, delta, kt_o_omega, chi, forces, 0, nmd.tetrahedra.size() / 4);

    for(int next_idx : nmd.bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : nmd.top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
}

//Helper function for computation of restoring forces when the network vertices
//belong to a set that is disjoint from the mesh vertices
void compute_forces_disjoint(NetworkMeshData &nmd,  double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces){

    cholmod_dense *net_forces, *net_contribution;
    void *net_force_values, *gel_force_values;
    double unity[2] = {1, 0}, goose_egg[2] = {0, 0};

    cblas_dscal(forces.size(), 0, &forces[0], 1);

    gel_forces_disjoint(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, nmd.reference_points, nmd.vertex_ranges, nmd.contained_vertices, gmod, delta, kt_o_omega, chi, forces, 0, nmd.tetrahedra.size() / 4);

    //cout << "Gel force magnitude: " << cblas_dnrm2(nmd.points.size()-3*nmd.num_net_verts, &forces[nmd.num_net_verts*3], 1) << endl;

    if(ks > 0){
        stretching_forces(nmd.points, nmd.edges, nmd.lengths, ks, forces, 0, nmd.edges.size()/2);
    }

    //cout << "RMS stretching force: " << cblas_dnrm2(nmd.num_net_verts*3, &forces[0], 1) / sqrt(3.*nmd.num_net_verts) << endl;

    if(kb > 0){
        bending_forces(nmd.points, nmd.triples, nmd.angles, kb, forces, 0, nmd.triples.size()/3);
    }

    //Project network forces onto the forces on the appropriate mesh vertices
    net_forces = cholmod_l_zeros(3*nmd.num_net_verts, 1, CHOLMOD_DOUBLE+CHOLMOD_REAL, &(nmd.common));
    net_contribution = cholmod_l_zeros(nmd.points.size() - 3*nmd.num_net_verts, 1, CHOLMOD_DOUBLE+CHOLMOD_REAL, &(nmd.common));
    net_force_values = net_forces->x;
    cblas_dcopy(3*nmd.num_net_verts, &forces[0], 1, (double *) net_force_values, 1);
    //cout << "CHOLMOD network force magnitude: " << cblas_dnrm2(3*nmd.num_net_verts, (double *) net_force_values, 1) << endl;
    //cout << "Sparse matrix Froebenius norm: " << cblas_ddot(nmd.force_mapping->nzmax, (double *) nmd.force_mapping->x, 1, (double *) nmd.force_mapping->x, 1) << endl;
    cholmod_l_sdmult(nmd.force_mapping, 0, unity, goose_egg, net_forces, net_contribution, &(nmd.common));
    gel_force_values = net_contribution->x;
    //cout << "Network contribution magnitude: " << cblas_dnrm2(nmd.points.size()-3*nmd.num_net_verts, (double *) gel_force_values, 1) << endl;
    cblas_daxpy(nmd.points.size()-3*nmd.num_net_verts, 1., (double *) gel_force_values, 1, &forces[3*nmd.num_net_verts], 1);
    //cout << "Force magnitude after addition: " << cblas_dnrm2(nmd.points.size()-3*nmd.num_net_verts, &forces[nmd.num_net_verts*3], 1) << endl;

    cholmod_l_free_dense(&net_forces, &(nmd.common));
    cholmod_l_free_dense(&net_contribution, &(nmd.common));

    for(int next_idx : nmd.bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : nmd.top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
}

//Helper functions to handle parallel force computation
void compute_forces_unified_parallel(NetworkMeshData &nmd, double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments){

    int iter, num_dof;

    num_dof = forces.size();

    cblas_dscal(num_dof, 0, &forces[0], 1);
    for(iter = 0; iter < force_sets.size(); iter++){
        cblas_dscal(num_dof, 0, &force_sets[iter][0], 1);
    }

    tbb::task_group g;

    if(ks > 0){
        for(iter = 0; iter < stretch_assignments.size() - 1; iter++){
            g.run([nmd, ks, &force_sets, stretch_assignments, iter]{
                stretching_forces(nmd.points, nmd.edges, nmd.lengths, ks, force_sets[iter], stretch_assignments[iter], stretch_assignments[iter+1]);
            });
        }
        g.wait();
    }

    if(kb > 0){
        for(iter = 0; iter < bend_assignments.size() - 1; iter++){
            g.run([nmd, kb, &force_sets, bend_assignments, iter]{
                bending_forces(nmd.points, nmd.triples, nmd.angles, kb, force_sets[iter], bend_assignments[iter], bend_assignments[iter+1]);
            });
        }
        g.wait();
    }

    if(nmd.tetrahedra.size()){
        for(iter = 0; iter < gel_assignments.size() - 1; iter++){
            g.run([nmd, gmod, delta, kt_o_omega, chi, &force_sets, gel_assignments, iter]{
                gel_forces(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, gmod, delta, kt_o_omega, chi, force_sets[iter], gel_assignments[iter], gel_assignments[iter+1]);
            });
        }
        g.wait();
    }

    for(iter = 0; iter < force_sets.size(); iter++){
        cblas_daxpy(num_dof, 1., &force_sets[iter][0], 1, &forces[0], 1);
    }

    for(int next_idx : nmd.bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : nmd.top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
}

void compute_forces_disjoint_parallel(NetworkMeshData &nmd,  double ks, double kb, double gmod, double delta, double kt_o_omega, double chi, vector<double> &forces, vector<vector<double>> &force_sets, vector<int> stretch_assignments, vector<int> bend_assignments, vector<int> gel_assignments){

    int iter, num_dof;
    cholmod_dense *net_forces, *net_contribution;
    void *net_force_values, *gel_force_values;
    cblas_dscal(forces.size(), 0, &forces[0], 1);
    double unity[2] = {1, 0};
    tbb::task_group g;

    num_dof = forces.size();

    cblas_dscal(num_dof, 0, &forces[0], 1);
    for(iter = 0; iter < force_sets.size(); iter++){
        cblas_dscal(num_dof, 0, &force_sets[iter][0], 1);
    }

    gel_forces_disjoint(nmd.points, nmd.tetrahedra, nmd.inv_mats, nmd.volumes, nmd.reference_points, nmd.vertex_ranges, nmd.contained_vertices, gmod, delta, kt_o_omega, chi, forces, 0, nmd.tetrahedra.size() / 4);

    if(ks > 0){
        for(iter = 0; iter < stretch_assignments.size() - 1; iter++){
            g.run([nmd, ks, &force_sets, stretch_assignments, iter]{
                stretching_forces(nmd.points, nmd.edges, nmd.lengths, ks, force_sets[iter], stretch_assignments[iter], stretch_assignments[iter+1]);
            });
        }
        g.wait();
    }

    if(kb > 0){
        for(iter = 0; iter < bend_assignments.size() - 1; iter++){
            g.run([nmd, kb, &force_sets, bend_assignments, iter]{
                bending_forces(nmd.points, nmd.triples, nmd.angles, kb, force_sets[iter], bend_assignments[iter], bend_assignments[iter+1]);
            });
        }
        g.wait();
    }

    //Project network forces onto the forces on the appropriate mesh vertices
    net_forces = cholmod_l_zeros(3*nmd.num_net_verts, 1, CHOLMOD_DOUBLE+CHOLMOD_REAL, &(nmd.common));
    net_contribution = cholmod_l_zeros(nmd.points.size() - 3*nmd.num_net_verts, 1, CHOLMOD_DOUBLE+CHOLMOD_REAL, &(nmd.common));
    net_force_values = net_forces->x;
    cblas_dcopy(3*nmd.num_net_verts, &forces[0], 1, (double *) net_force_values, 1);
    cholmod_l_sdmult(nmd.force_mapping, 0, unity, unity, net_forces, net_contribution, &(nmd.common));
    gel_force_values = net_contribution->x;
    cblas_daxpy(nmd.points.size()-3*nmd.num_net_verts, 1., (double *) gel_force_values, 1, &forces[3*nmd.num_net_verts], 1);

    cholmod_l_free_dense(&net_forces, &(nmd.common));
    cholmod_l_free_dense(&net_contribution, &(nmd.common));

    for(int next_idx : nmd.bottom){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
    for(int next_idx : nmd.top){
        forces[3*next_idx] = 0;
        forces[3*next_idx+1] = 0;
        forces[3*next_idx+2] = 0;
    }
}

bool parse_nmc_file(string file_name, NetworkMeshData &nmd){

    double minx = FLT_MAX, maxx = -FLT_MAX, miny = FLT_MAX, maxy = -FLT_MAX;
    double minz = FLT_MAX, maxz = -FLT_MAX;
    int iter, idx1, idx2, offset;
    Vec3D p1, p2;

    //Read compressed, binary data from the nmc file
    if(nmd.type == nmc_type::unified){
        if(! extract_network_mesh_complex((char *) file_name.c_str(), nmd.points, nmd.edges, nmd.triples, nmd.angles, nmd.tetrahedra)){
            cerr << "The network-mesh extraction failed.\n";
            return false;
        }
        offset = 0;
    }
    else{
        if(! extract_disjoint_network_mesh_complex((char *) file_name.c_str(), nmd.points, nmd.num_net_verts, nmd.edges, nmd.triples, nmd.angles, nmd.tetrahedra, nmd.vertex_ranges, nmd.contained_vertices)){
            cerr << "The network-mesh extraction failed.\n";
            return false;
        }
        offset = 3*nmd.num_net_verts;
    }

    //Find the extremes along each direction and identify top and bottom points
    for(iter = 0; iter < nmd.points.size(); iter += 3){
        if(nmd.points[iter] < minx) minx = nmd.points[iter];
        if(nmd.points[iter+1] < miny) miny = nmd.points[iter+1];
        if(nmd.points[iter+2] < minz) minz = nmd.points[iter+2];
        if(nmd.points[iter] > maxx) maxx = nmd.points[iter];
        if(nmd.points[iter+1] > maxy) maxy = nmd.points[iter+1];
        if(nmd.points[iter+2] > maxz) maxz = nmd.points[iter+2];
    }

    nmd.bounds.push_back(minx);
    nmd.bounds.push_back(maxx);
    nmd.bounds.push_back(miny);
    nmd.bounds.push_back(maxy);
    nmd.bounds.push_back(minz);
    nmd.bounds.push_back(maxz);

    for(iter = offset + 2; iter < nmd.points.size(); iter += 3){
        if(nmd.points[iter] < minz + TOL) nmd.bottom.insert(iter / 3);
        if(nmd.points[iter] > maxz - TOL) nmd.top.insert(iter / 3);
    }

    //Find resting edge lengths
    for(iter = 0; iter < nmd.edges.size(); iter += 2){
        idx1 = nmd.edges[iter];
        idx2 = nmd.edges[iter+1];
        assign(p1, nmd.points, idx1);
        assign(p2, nmd.points, idx2);
        nmd.lengths.push_back(distance(p1, p2));
    }

    return true;
}

//Helper function to update arrays of row and column indices within triplet
//sparse matrix data structures
void assign_index(void *index_list, int itype, int index, int value){
    if(itype == CHOLMOD_LONG){
        ((SuiteSparse_long *) index_list)[index] = value;
    }
    else{
        ((int *) index_list)[index] = value;
    }
}

//Helper function to update an array of matrix elements within a triplet sparse
//matrix structure
void assign_mat_elem(void *elem_list, int index, double value){
    ((double *) elem_list)[index] = value;
}

bool compute_network_mesh_map(NetworkMeshData &nmd){

    vector<double> inv_mat;
    int rows, cols, nzmax, pos, tet_iter, net_pt_iter, vert_iter, itype;
    int net_pt_idx, vert_idx;
    double coeff, aug_coord[4];

    //Triplet and CSC forms of the mapping from network to mesh vertex forces
    cholmod_triplet *triplet_map;

    //Pointers to the row, column and non-zero matrix element data in the
    //triplet representation
    void *map_i, *map_j, *map_x;

    //Ascertain dimensions of and allocate triplet representation
    cols = nmd.num_net_verts*3;
    rows = nmd.points.size() - cols;
    nzmax = 12*nmd.num_net_verts;
    triplet_map = cholmod_l_allocate_triplet(rows, cols, nzmax, 0, CHOLMOD_DOUBLE+CHOLMOD_REAL, &(nmd.common));
    triplet_map->nnz = nzmax;
    if(nmd.common.status < CHOLMOD_OK){
        cerr << "The triplet force map representation could not be allocated.\n";
        cholmod_l_free_triplet(&triplet_map, &(nmd.common));
        return false;
    }

    map_i = triplet_map->i;
    map_j = triplet_map->j;
    map_x = triplet_map->x;
    itype = triplet_map->itype;

    //The fourth element of the augmented coordinate is always 1
    aug_coord[3] = 1.;

    pos = 0;

    //For each network vertex within a given mesh element, compute the maping
    //from the network vertex's force to the forces on the element vertices
    for(tet_iter = 0; tet_iter < nmd.tetrahedra.size() / 4; tet_iter ++){
        inv_mat.assign(nmd.inv_mats.begin()+16*tet_iter, nmd.inv_mats.begin()+16*(tet_iter+1));

        for(net_pt_iter = nmd.vertex_ranges[tet_iter]; net_pt_iter != nmd.vertex_ranges[tet_iter+1]; net_pt_iter++){
            net_pt_idx = nmd.contained_vertices[net_pt_iter];
            aug_coord[0] = nmd.points[net_pt_idx*3];
            aug_coord[1] = nmd.points[net_pt_idx*3+1];
            aug_coord[2] = nmd.points[net_pt_idx*3+2];

            for(vert_iter = 0; vert_iter < 4; vert_iter ++){
                coeff = cblas_ddot(4, &inv_mat[4*vert_iter], 1, aug_coord, 1);
                vert_idx = nmd.tetrahedra[4*tet_iter+vert_iter];
                assign_index(map_i, itype, pos, 3*vert_idx-cols); 
                assign_index(map_j, itype, pos, 3*net_pt_idx);
                assign_mat_elem(map_x, pos, coeff);
                assign_index(map_i, itype, pos+1, 3*vert_idx+1-cols); 
                assign_index(map_j, itype, pos+1, 3*net_pt_idx+1);
                assign_mat_elem(map_x, pos+1, coeff);
                assign_index(map_i, itype, pos+2, 3*vert_idx+2-cols);
                assign_index(map_j, itype, pos+2, 3*net_pt_idx+2);
                assign_mat_elem(map_x, pos+2, coeff);
                pos += 3;
            }
        }
    }

    //Convert the triple representation to a CSC representation
    nmd.force_mapping = cholmod_l_triplet_to_sparse(triplet_map, nzmax, &(nmd.common));
    cholmod_l_free_triplet(&triplet_map, &(nmd.common));

    return true;
}
