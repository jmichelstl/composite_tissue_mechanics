#include "geometric_primitives.hpp"
#include <cmath>
#include <time.h>
#include <tbb.h>
#include <functional>
#include <numeric>
#include "random_point_processes.hpp"
#include <random>
#include <voro++.hh>
#include <iostream>

using std::vector;

//Maximum permissible number of attempts
#define MAX_ATTEMPTS 100

//Initial number of points for which to allocate memory in each box of the
//Voro++ grid
#define PTS_PER_BOX 8

#define min(a, b) a < b ? a : b; 

#define max(a, b) a > b ? a : b; 

//Utility for parallel summation
template <typename T> T sum_reduce( const T* first, const T* last, T identity ) {
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

//Given a point, find the index of the box containing it within a list a grid
inline int get_grid_idx(Vec3D pt, Bounds3D bounds, double grid_len, int nx, int ny, int nz){

    double x_idx, y_idx, z_idx;

    x_idx = (int) floor((pt.x - bounds.min_x()) / grid_len);
    y_idx = (int) floor((pt.y - bounds.min_y()) / grid_len);
    z_idx = (int) floor((pt.z - bounds.min_z()) / grid_len);

    return nx*ny*z_idx + nx*y_idx + x_idx;
}

/*
 * Generate a Poisson sphere packing.
 */
vector<double> poisson_sphere_packing(Bounds3D bounds, double cutoff){

    vector<Vec3D> points, active_points;
    vector<int> indices;
    double minx, maxx, miny, maxy, minz, maxz;
    double grid_len, r_x, r_y, r_z, r_r, r_cos, r_sin, r_phi, dx, dy, dz;
    double cutoff_sq = cutoff*cutoff, xrange, yrange, zrange, expect_val;
    int nx, ny, nz, rand_idx, x_idx, y_idx, z_idx, grid_idx, attempts, i, j, k;
    int min_x_iter, max_x_iter, min_y_iter, max_y_iter, min_z_iter, max_z_iter;
    int replentish_attempts;
    time_t now = time(0);
    std::mt19937 gen{static_cast<uint32_t>(now)};
    std::uniform_real_distribution<double> rdist(0.0, 1.0);
    std::uniform_int_distribution<int> idist(0, 0);
    bool success;
    Vec3D active_pt, new_pt;
    vector<double> coordinates;

    //Set up the grid for indexing points
    minx = bounds.min_x();
    maxx = bounds.max_x();
    miny = bounds.min_y();
    maxy = bounds.max_y();
    minz = bounds.min_z();
    maxz = bounds.max_z();
    xrange = maxx - minx;
    yrange = maxy - miny;
    zrange = maxz - minz;
    grid_len = cutoff / sqrt(3.);
    nx = (int) ceil((maxx - minx) / grid_len);
    ny = (int) ceil((maxy - miny) / grid_len);
    nz = (int) ceil((maxz - minz) / grid_len);
    indices.assign(nx*ny*nz, -1);

    //Compute the expected number of times a point would have to randomly drawn
    //such that it falls in a cube of side length cutoff. Use this to determine
    //the number of times to attempt to add a new active point before aborting.
    expect_val = xrange*yrange*zrange / (cutoff*cutoff*cutoff);
    replentish_attempts = (int) (expect_val + 2*sqrt(expect_val));

    //Create an inaugural point
    r_x = rdist(gen)*xrange + minx;
    r_y = rdist(gen)*yrange + miny;
    r_z = rdist(gen)*zrange + minz;
    points.push_back(Vec3D(r_x, r_y, r_z));
    active_points.push_back(points[0]);
    rand_idx = get_grid_idx(points[0], bounds, grid_len, nx, ny, nz);
    indices[rand_idx] = 0;

    //Until the active set is empty, choose a point from the active set, and
    //generate another point by generating a random displacement. Determine
    //whether any points are too near the proposed point. If not, add the
    //point, and if so, try again. If a suitable point cannot be created after
    //MAX_ATTEMPTS attempts, give up and remove the starting point from the
    //active set.
    do{

        //Reset the range of the integer distribution according to the current
	    //number of active points, and draw a random index.
	    {
	        std::uniform_int_distribution<int>::param_type new_par(0, active_points.size()-1);
	        idist.param(new_par);
        }
	    rand_idx = idist(gen);
	    attempts = 0;
	    active_pt = points[rand_idx];

        do{
            success = true;

	        do{
                r_r = cutoff*(rdist(gen) + 1);
                r_cos = 2*rdist(gen) - 1;
                r_sin = sqrt(1 - r_cos*r_cos);
                r_phi = 2*M_PI*rdist(gen);

	            new_pt.x = active_pt.x + r_r*r_sin*cos(r_phi);
	            new_pt.y = active_pt.y + r_r*r_sin*sin(r_phi);
		        new_pt.z = active_pt.z + r_r*r_cos;
	            x_idx = (int) floor((new_pt.x - minx) / grid_len);
                y_idx = (int) floor((new_pt.y - miny) / grid_len);
                z_idx = (int) floor((new_pt.z - minz) / grid_len);
            } while(! bounds.in_bounds(new_pt));

	        x_idx = (int) floor((new_pt.x - minx) / grid_len);
            y_idx = (int) floor((new_pt.y - miny) / grid_len);
            z_idx = (int) floor((new_pt.z - minz) / grid_len);
            if(indices[z_idx*nx*ny + y_idx*nx + x_idx] >= 0){
                success = false;
                continue;
            }
            min_x_iter = max(x_idx - 2, 0);
            max_x_iter = min(x_idx + 2, nx-1);
            min_y_iter = max(y_idx - 2, 0);
            max_y_iter = min(y_idx + 2, ny-1);
            min_z_iter = max(z_idx - 2, 0);
            max_z_iter = min(z_idx + 2, nz-1);

            for(i = min_x_iter; i <= max_x_iter; i++){
	            for(j = min_y_iter; j <= max_y_iter; j++){
                    for(k = min_z_iter; k <= max_z_iter; k++){
                        grid_idx = k*nx*ny + j*nx + i;
			            if(indices[grid_idx] > -1){
                            if(distance_sq(new_pt, points[indices[grid_idx]]) < cutoff_sq){
                                success = false;
				                break;
                            }
                        }
                    }
		            if(! success) break;
                }
		        if(! success) break;
            }
            attempts ++;
        } while(! success && attempts < MAX_ATTEMPTS);

        if(! success){
            active_points.erase(active_points.begin() + rand_idx);
        }

        else{
            indices[z_idx*nx*ny + y_idx*nx + x_idx] = points.size();
	        points.push_back(new_pt);
	        active_points.push_back(new_pt);
        }

        //If there are no more active points, try to randomly place another
	    //point. If this fails, the volume is deemed full.
	    if(active_points.size() > 0) continue;

        attempts = 0;
	    do{
            success = true;

            new_pt.x = minx + rdist(gen)*xrange;
            new_pt.y = miny + rdist(gen)*yrange;
            new_pt.z = minz + rdist(gen)*zrange;
	        x_idx = (int) floor((new_pt.x - minx) / grid_len);
            y_idx = (int) floor((new_pt.y - miny) / grid_len);
            z_idx = (int) floor((new_pt.z - minz) / grid_len);
            min_x_iter = max(x_idx - 2, 0);
            max_x_iter = min(x_idx + 2, nx-1);
            min_y_iter = max(y_idx - 2, 0);
            max_y_iter = min(y_idx + 2, ny-1);
            min_z_iter = max(z_idx - 2, 0);
            max_z_iter = min(z_idx + 2, nz-1);

	        for(i = min_x_iter; i <= max_x_iter; i++){
	            for(j = min_y_iter; j <= max_y_iter; j++){
                    for(k = min_z_iter; k <= max_z_iter; k++){
                        grid_idx = k*nx*ny + j*nx + i;
                        if(indices[grid_idx] > -1){
                            if(distance_sq(new_pt, points[indices[grid_idx]]) < cutoff_sq){
                                success = false;
                                break;
                            }
                        }
                    }
	                if(! success) break;
                }
	            if(! success) break;
            }

            if(success){
                indices[z_idx*ny*ny + y_idx*nx + x_idx] = points.size();
                points.push_back(new_pt);
                active_points.push_back(new_pt);
            }
        } while (! success && ++attempts < replentish_attempts);

    }while(active_points.size() > 0);

    for(Vec3D pt : points){
        coordinates.push_back(pt.x);
        coordinates.push_back(pt.y);
        coordinates.push_back(pt.z);
    }

    return coordinates;
}

/*Result of integrating the Lloyd noise over a tetrahedron
In the general case, for a centroid rc and tetrahedron vertices r1, r2, r3
and r4, the result has the form
    r1_i*r1_i + r2_i*r2_i + r3_i*r3_i + r4_i*r4_i) / 60
    + (r1_i*r2_i + r1_i*r3_i + r1_i*r4_i + r2_i*r3_i + r2_i*r4_i + r3_i*r4_i)/60
    - rc_i * (r1_i + r2_i + r3_i + r4_i) / 12 + rc_i*rc_i / 6,

where summation over repeated Cartesian indices is implied. Here, it is
assumed that one point in the tetrahedron is the centroid, and that the centroid
is at the origin.
*/
double lloyd_integral(double r1, double r2, double r3){

    double integral;

    integral = (r1*r1 + r2*r2 + r3*r3) / 60;
    integral += (r1*r2 + r1*r3 + r2*r3) / 60;

    return integral; 
}

//Create a boundary mesh for a polyhedron, partition into tetrahedra, and
//compute the contribution to the Lloyd noise from each tetrahedron.
void calc_lloyd_noise(int idx, vector<Vec3D> centroids, vector<vector<double>> vertex_sets, vector<vector<int>> face_sets, vector<double> &noise_list){

    int i, j, face_iter = 0, vnum;
    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, tet_noise;
    Vec3D centroid, v1, v2, v3, p1, p2, mid1, mid2, cross_prod;
    vector<double> vertices = vertex_sets[idx];
    vector<int> faces = face_sets[idx];

    noise_list[idx] = 0;
    copy(centroids[idx], centroid);

    //Loop over faces. Partition each face into triangles. From each triangle,
    //construct a tetrahedron whose fourth point is the cell centroid, and
    //determine the contribution to the Lloyd noise of that tetrahedron.
    do{

        vnum = faces[face_iter];

        //Find an interior point for the present face
        assign(p1, vertices, faces[face_iter + 1]);
        assign(p2, vertices, faces[face_iter + 2]);
        get_midpoint(p1, p2, mid1);
        assign(p1, vertices, faces[face_iter + vnum/2]);
        assign(p2, vertices, faces[face_iter + vnum/2 + 1]);
        get_midpoint(p1, p2, mid2);
        get_midpoint(mid1, mid2, v1);
        difference(v1, centroid, v1); 

        for(i = 1; i < faces[face_iter]; i++){
            assign(v2, vertices, faces[face_iter+i]);
            difference(v2, centroid, v2);
            assign(v3, vertices, faces[face_iter + i%vnum + 1]);
            difference(v3, centroid, v3);
            tet_noise = lloyd_integral(v1.x, v2.x, v3.x);
            tet_noise += lloyd_integral(v1.y, v2.y, v3.y);
            tet_noise += lloyd_integral(v1.z, v2.z, v3.z);
            cross(v1, v2, cross_prod);
            noise_list[idx] += abs(dot(cross_prod, v3))*tet_noise;
        }

        face_iter += vnum + 1;
    } while(face_iter < faces.size());
}

//Given a Voronoi container with a point set, find numbers of faces, centroids,
//vertices, and facets for all cells.
void compute_cell_data(vector<double> coordinates, Bounds3D bounds, int nx, int ny, int nz, vector<Vec3D> &centroids, vector<vector<double>> &vertex_sets, vector<vector<int>> &face_sets){

    voro::voronoicell cell;
    double cx, cy, cz, x, y, z;
    vector<double> vertices;
    vector<int> faces;

    centroids.clear();
    vertex_sets.clear();
    face_sets.clear();

    voro::container_periodic con(bounds.max_x(), 0, bounds.max_y(), 0, 0, bounds.max_z(),nx,ny,nz,PTS_PER_BOX);
    for(int iter = 0; iter < coordinates.size(); iter += 3){
        con.put(iter/3, coordinates[iter], coordinates[iter+1], coordinates[iter+2]);
    }
    voro::c_loop_all_periodic cloop(con);

    if(cloop.start()){
        do if(con.compute_cell(cell, cloop)){
            cloop.pos(x, y, z);
	        cell.centroid(cx, cy, cz);
	        centroids.push_back(Vec3D(cx+x, cy+y, cz+z));
            cell.vertices(x, y, z, vertices);
	        vertex_sets.push_back(vertices);
	        cell.face_vertices(faces);
	        face_sets.push_back(faces);
        } while (cloop.inc());
    }
}

vector<double> compute_hyperuniform_packing(Bounds3D bounds, double density, double conv_tol, int max_iter){

    double curr_noise, prev_noise, rel_diff, grid_len, cutoff;
    int iter = 0, nx, ny, nz;
    vector<double> coordinates, noises;
    vector<double> vertices;
    vector<Vec3D> centroids;
    vector<vector<double>> vertex_sets;
    vector<vector<int>> face_sets;
    double *begin;

    grid_len = pow(density, -1./3);
    nx = (int) ceil(bounds.xrange() / grid_len);
    ny = (int) ceil(bounds.yrange() / grid_len);
    nz = (int) ceil(bounds.zrange() / grid_len);

    //Given a desired density, use an empirically determined value to convert
    //to a minimum separation between points
    cutoff = pow(DENSITY_TO_LENGTH / density, 1./3.);

    //Obtain the original Poisson packing
    coordinates = poisson_sphere_packing(bounds, cutoff);

    noises.assign(coordinates.size() / 3, 0);

    //Find the original Voronoi tessellation and the accompanying Lloyd noise
    compute_cell_data(coordinates, bounds, nx, ny, nz, centroids, vertex_sets, face_sets);

    //Compute noises and sum them in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>(0, noises.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i){
                calc_lloyd_noise(i,centroids,vertex_sets,face_sets,noises);
            }
        }
    );

    begin = &noises[0];
    prev_noise = sum_reduce(begin, begin+noises.size(), 0.);

    //Perform Lloyd moves until either the convergence tolerance is satisfied
    //or the iteration limit is reached.
    do {

        //Perform a  Lloyd move, replacing each point by the cenroid of the
        //Voronoi cell it generates.
        tbb::parallel_for(tbb::blocked_range<size_t>(0, noises.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    coordinates[3*i] = centroids[i].x;
                    if(centroids[i].x < bounds.min_x()){
                        coordinates[3*i] += bounds.xrange();
                    }
                    else if(centroids[i].x > bounds.max_x()){
                        coordinates[3*i] -= bounds.xrange();
                    }
                    coordinates[3*i+1] = centroids[i].y;
                    if(centroids[i].y < bounds.min_y()){
                        coordinates[3*i+1] += bounds.yrange();
                    }
                    else if(centroids[i].y > bounds.max_y()){
                        coordinates[3*i+1] -= bounds.yrange();
                    }
                    coordinates[3*i+2] = centroids[i].z;
                    if(centroids[i].z < bounds.min_z()){
                        coordinates[3*i+2] += bounds.zrange();
                    }
                    else if(centroids[i].z > bounds.max_z()){
                        coordinates[3*i+2] -= bounds.zrange();
                    }
                }
            }
        );

	   //Compute the new cell data and Lloyd noises
       compute_cell_data(coordinates, bounds, nx, ny, nz, centroids, vertex_sets, face_sets);

        //Compute noises and sum them in parallel
        tbb::parallel_for(tbb::blocked_range<size_t>(0, noises.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    calc_lloyd_noise(i,centroids,vertex_sets,face_sets,noises);
                }
            }
        );

        begin = &noises[0];
        curr_noise = sum_reduce(begin, begin+noises.size(), 0.);
	    rel_diff = abs((curr_noise - prev_noise) / prev_noise);
	    prev_noise = curr_noise;

    } while(rel_diff > conv_tol && ++iter < max_iter);

    return coordinates;
}
