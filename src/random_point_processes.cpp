#include "geometric_primitives.hpp"
#include <cmath>
#include <time.h>
#include <tbb.h>
#include <functional>
#include <numeric>
#include "random_point_processes.hpp"
#include <random>
#include <voro++.hh>
#include <tetgen.h>
#include <iostream>

using std::vector;

using namespace voro;

//Maximum permissible number of attempts
#define MAX_ATTEMPTS 75

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
    double grid_len, r_x, r_y, r_z, r_r, r_cos, r_sin, r_phi, dx, dy, dz;
    double cutoff_sq = cutoff*cutoff, xrange, yrange, zrange;
    double minx, maxx, miny, maxy, minz, maxz;
    int nx, ny, nz, rand_idx, x_idx, y_idx, z_idx, grid_idx, attempts, i, j, k;
    int min_x_iter, max_x_iter, min_y_iter, max_y_iter, min_z_iter, max_z_iter;
    time_t now = time(0);
    std::mt19937 gen{static_cast<uint32_t>(now)};
    std::uniform_real_distribution<double> rdist(0.0, 1.0);
    std::uniform_int_distribution<int> idist(0, 0);
    bool success;
    Vec3D active_pt, new_pt;
    vector<double> coordinates;

    minx = bounds.min_x();
    maxx = bounds.max_x();
    miny = bounds.min_y();
    maxy = bounds.max_y();
    minz = bounds.min_z();
    maxz = bounds.max_z();

    std::cout << "Volume: " << bounds.get_vol() << std::endl;
    std::cout << "Cutoff: " << cutoff << std::endl;

    //Set up the grid for indexing points
    xrange = bounds.xrange();
    yrange = bounds.yrange();
    zrange = bounds.zrange();
    grid_len = cutoff / sqrt(3.);
    nx = (int) ceil(xrange / grid_len);
    ny = (int) ceil(yrange / grid_len);
    nz = (int) ceil(zrange / grid_len);
    indices.assign(nx*ny*nz, -1);

    //Create an inaugural point
    r_x = rdist(gen)*xrange + minx;
    r_y = rdist(gen)*yrange + miny;
    r_z = rdist(gen)*zrange + minz;
    points.push_back(Vec3D(r_x, r_y, r_z));
    active_points.push_back(points[0]);
    indices[get_grid_idx(points[0], bounds, grid_len, nx, ny, nz)] = 0;

    //Until the active set is empty, choose a point from the active set, and
    //generate another point by generating a random displacement. Determine
    //whether any points are too near the proposed point. If not, add the
    //point, and if so, try again. If a suitable point cannot be created after
    //MAX_ATTEMPTS attempts, give up and remove the starting point from the
    //active set.
    do{

        //Reset the range of the integer distribution according to the current
        //number of active points, and draw a random index.
	    std::uniform_int_distribution<int>::param_type new_par(0, active_points.size()-1);
	    idist.param(new_par);
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
            } while(! bounds.in_bounds(new_pt));

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

            new_pt.x = rdist(gen)*xrange + minx;
            new_pt.y = rdist(gen)*yrange + miny;
            new_pt.z = rdist(gen)*zrange + minz;
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
                indices[z_idx*nx*ny + y_idx*nx + x_idx] = points.size();
                points.push_back(new_pt);
                active_points.push_back(new_pt);
            }
        } while (! success && ++attempts < MAX_ATTEMPTS);

    }while(active_points.size() > 0);

    for(Vec3D pt : points){
        coordinates.push_back(pt.x);
        coordinates.push_back(pt.y);
        coordinates.push_back(pt.z);
    }

    return coordinates;
}

//Result of integrating the Lloyd noise over a tetrahedron
double lloyd_integral(double r_c, double r1, double r2, double r3, double r4){

    double integral;

    integral = (r1*r1 + r2*r2 + r3*r3 + r4*r4) / 60;
    integral += (r1*r2 + r1*r3 + r1*r4 + r2*r3 + r2*r4 + r3*r4) / 60;
    integral -= r_c * (r1 + r2 + r3 + r4) / 12;
    integral += r_c*r_c / 6;

    return integral; 
}

void assign_point(double *plist, int idx, double &x, double &y, double &z){

    x = plist[3*idx];
    y = plist[3*idx+1];
    z = plist[3*idx+2];
}

//Create a boundary mesh for a polyhedron, partition into tetrahedra, and
//compute the contribution to the Lloyd noise from each tetrahedron.
void calc_lloyd_noise(int idx, vector<int> face_nums, vector<Vec3D> centroids, vector<vector<double>> vertex_sets, vector<vector<int>> face_sets, vector<double> &noise_list){

    int i, j, face_iter = 0;
    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, det;
    tetgenio in, out;
    tetgenio::facet *f;
    tetgenio::polygon *p;
    const char *flags = "pq1.414";
    in.firstnumber = 0;

    Vec3D centroid = centroids[idx];
    vector<double> vertices = vertex_sets[idx];
    vector<int> faces = face_sets[idx];

    //Copy over points
    in.numberofpoints = vertices.size() / 3;
    in.pointlist = new double[in.numberofpoints * 3];
    for(i = 0; i < vertices.size(); i ++){
        in.pointlist[i] = vertices[i];
    }

    //Create faces
    in.numberoffacets = face_nums[idx];
    in.facetmarkerlist = NULL;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    for(i = 0; i < in.numberoffacets; i++){
        f = &in.facetlist[i];
	    f->numberofpolygons = 1;
	    f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
	    f->numberofholes = 0;
	    f->holelist = NULL;
	    p = &f->polygonlist[0];
	    p->numberofvertices = vertices[face_iter];
        p->vertexlist = new int[p->numberofvertices];

	    for(j = 1; j <= p->numberofvertices; j++){
            p->vertexlist[j] = faces[face_iter + j];
        }
	    face_iter += faces[face_iter] + 1;
    }

    //Create the tetrahedralization
    tetrahedralize((char *) flags, &in, &out);

    //Loop over tetrahedra and compute the Lloyd noise contribution of each
    for(i = 0; i < out.numberoftetrahedra; i += 4){
        assign_point(out.pointlist, out.tetrahedronlist[4*i],x1,y1,z1);
        assign_point(out.pointlist, out.tetrahedronlist[4*i+1],x2,y2,z2);
        assign_point(out.pointlist, out.tetrahedronlist[4*i+2],x3,y3,z3);
        assign_point(out.pointlist, out.tetrahedronlist[4*i+3],x4,y4,z4);
        det = (x2-x1)*((y3-y1)*(z4-z1) - (y4-y1)*(z3-z1));
        det += (x3-x1)*((y4-y1)*(z2-z1) - (y2-y1)*(z4-z1));
        det += (x4-x1)*((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1));
        noise_list[idx] = lloyd_integral(centroid.x, x1, x2, x3, x4);
        noise_list[idx] += lloyd_integral(centroid.y, y1, y2, y3, y4);
        noise_list[idx] += lloyd_integral(centroid.z, z1, z2, z3, z4);
        noise_list[idx] *= abs(det);
    }
}

//Given a Voronoi container with a point set, find numbers of faces, centroids,
//vertices, and facets for all cells.
void compute_cell_data(container con, vector<int> &face_nums, vector<Vec3D> &centroids, vector<vector<double>> &vertex_sets, vector<vector<int>> &face_sets){

    c_loop_all cloop(con);
    voronoicell cell;
    double x, y, z;
    vector<double> vertices;
    vector<int> faces;

    face_nums.clear();
    centroids.clear();
    vertex_sets.clear();
    face_sets.clear();

    if(cloop.start()){
        do if(con.compute_cell(cell, cloop)){
            face_nums.push_back(cell.number_of_faces());
	        cell.centroid(x, y, z);
	        centroids.push_back(Vec3D(x, y, z));
            cell.vertices(vertices);
	        vertex_sets.push_back(vertices);
	        cell.face_vertices(faces);
	        face_sets.push_back(faces);
        } while (cloop.inc());
    }
}

vector<double> compute_hyperuniform_packing(Bounds3D bounds, double density, double conv_tol, int max_iter){

    double curr_noise, prev_noise, rel_diff, grid_len;
    double minx, maxx, miny, maxy, minz, maxz, cutoff;
    int iter = 0, nx, ny, nz;
    vector<double> coordinates, noises;
    vector<int> faces;
    vector<double> vertices;
    vector<int> face_nums;
    vector<Vec3D> centroids;
    vector<vector<double>> vertex_sets;
    vector<vector<int>> face_sets;
    double *begin;

    minx = bounds.min_x();
    maxx = bounds.max_x();
    miny = bounds.min_y();
    maxy = bounds.max_y();
    minz = bounds.min_z();
    maxz = bounds.max_z();

    grid_len = pow(density, -1./3);
    nx = (int) ceil(bounds.xrange() / grid_len);
    nz = (int) ceil(bounds.yrange() / grid_len);
    nz = (int) ceil(bounds.zrange() / grid_len);

    //Given a desired density, use an empirically determined value to convert
    //to a minimum separation between points
    cutoff = DENSITY_TO_LENGTH / pow(density, 1/3);

    //Obtain the original Poisson packing
    coordinates = poisson_sphere_packing(bounds, cutoff);
    noises.assign(coordinates.size() / 3, 0);

    container con(minx,maxx,miny,maxy,minz,maxz,nx,ny,nz,false,false,false,PTS_PER_BOX);

    //Find the original Voronoi tessellation and the accompanying Lloyd noise
    for(int iter = 0; iter < coordinates.size(); iter += 3){
        con.put(iter/3, coordinates[iter], coordinates[iter+1], coordinates[iter+2]);
    }
    compute_cell_data(con, face_nums, centroids, vertex_sets, face_sets);

    //Compute noises and sum them in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>(0, noises.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i){
                calc_lloyd_noise(i, face_nums, centroids, vertex_sets, face_sets, noises);
            }
        }
    );
    begin = &noises[0];
    prev_noise = sum_reduce(begin, begin+noises.size(), 0.);

    //Perform a  Lloyd move, replacing each point by the cenroid of the
    //Voronoi cell it generates.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, noises.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i){
                coordinates[3*i] = centroids[i].x;
                coordinates[3*i+1] = centroids[i].y;
                coordinates[3*i+2] = centroids[i].z;
            }
        }
    );

    //Perform Lloyd moves until either the convergence tolerance is satisfied
    //or the iteration limit is reached.
    do {
        //Perform a  Lloyd move, replacing each point by the cenroid of the
        //Voronoi cell it generates.
        tbb::parallel_for(tbb::blocked_range<size_t>(0, noises.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    coordinates[3*i] = centroids[i].x;
                    coordinates[3*i+1] = centroids[i].y;
                    coordinates[3*i+2] = centroids[i].z;
                }
            }
        );

	    //Compute the new cell data and Lloyd noises
        for(int iter = 0; iter < coordinates.size(); iter += 3){
            con.put(iter/3, coordinates[iter], coordinates[iter+1], coordinates[iter+2]);
        }
        compute_cell_data(con, face_nums, centroids, vertex_sets, face_sets);

        //Compute noises and sum them in parallel
        tbb::parallel_for(tbb::blocked_range<size_t>(0, noises.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    calc_lloyd_noise(i, face_nums, centroids, vertex_sets, face_sets, noises);
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
