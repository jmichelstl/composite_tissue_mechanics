#include "network_mesh_complex.hpp"
#include "parse_lat3d_file.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_set>
#include <time.h>
#include <random>
#include <voro++.hh>
#include <gmsh.h>
#include <set>
#include <cfloat>
#include "nmbvh.hpp"
#include <metis.h>

#define PTS_PER_BOX 8

using namespace std;

//Default values for halting when making random point hyperuniform point sets
double hu_max_iter = 1e3;
double hu_conv_tol = 1e-6;

/*Struct to sort points in conter-clockwise order according to the projection in
  a plane of the difference between each point and a central point.
  Constructor arguments:
        -center, the common center
        -v1, one of two vectors spanning the plane
        -v2, the other vector spanning the plane
*/
struct ccw_sort{ 

    Vec3D center, v1, v2;

    ccw_sort(Vec3D my_center, Vec3D my_v1, Vec3D my_v2){
        center = my_center;
        v1 = my_v1; 
        v2 = my_v2;
    }

    bool operator() (Vec3D p1, Vec3D p2){
        double ang1, ang2;
        difference(p1, center, p1);
        difference(p2, center, p2);
        ang1 = atan2(dot(p1, v2), dot(p1, v1));
        if(ang1 < 0) ang1 += 2*M_PI;
        ang2 = atan2(dot(p2, v2), dot(p2, v1));
        if(ang2 < 0) ang2 += 2*M_PI;
        return ang1 < ang2;
    }
};

/*
Given a set of points within or on the surface of an axis-aligned rectangular
prism, place each point in each of fourteen classes to which it may belong:
    -Points on the interior of the face whose outward normal is along the -x 
    axis
    -Points on the edge of the face whose outward normal is along the -x axis
    -Points on the interior of the face whose outward normal along the the +x
    -Points on the edge of the face whose outward normal is along the +x axis
    -Points on the interior of the face whose outward normal is along the -y 
    axis
    -Points on the edge of the face whose outward normal is along the -y axis
    -Points on the interior of the face whose outward normal along the the +y
    axis
    -Points on the edge of the face whose outward normal is along the +y axis
    -Points on the interior of the face whose outward normal is along the -z 
    axis
    -Points on the edge of the face whose outward normal is along the -z axis
    -Points on the interior of the face whose outward normal along the the +z
    axis
    -Points on the edge of the face whose outward normal is along the +z axis
    -Points on the interior of the prism
*/
void categorize_points(double minx, double maxx, double miny, double maxy, double minz, double maxz, vector<double> in_points, vector<set<Vec3D>> &face_sets, vector<Vec3D> &interior){

    int iter;
    Vec3D point;
    set<Vec3D> minus_x_int, minus_x_edge, plus_x_int, plus_x_edge;
    set<Vec3D> minus_y_int, minus_y_edge, plus_y_int, plus_y_edge;
    set<Vec3D> minus_z_int, minus_z_edge, plus_z_int, plus_z_edge;
    bool on_face;

    for(iter = 0; iter < in_points.size(); iter += 3){

        point = Vec3D(in_points[iter], in_points[iter+1], in_points[iter+2]);
        on_face = false;

        if(point.x < minx + TOL){
            if(point.y < miny + TOL || point.y > maxy - TOL || point.z < minz + TOL || point.z > maxz - TOL) minus_x_edge.insert(point);
            else minus_x_int.insert(point);
            on_face = true;
        } 
        if(point.x > maxx -  TOL){
            if(point.y < miny + TOL || point.y > maxy - TOL || point.z < minz + TOL || point.z > maxz - TOL) plus_x_edge.insert(point);
            else plus_x_int.insert(point);
            on_face = true;
        } 
        if(point.y < miny + TOL){
            if(point.x < minx + TOL || point.x > maxx - TOL || point.z < minz + TOL || point.z > maxz - TOL) minus_y_edge.insert(point);
            else minus_y_int.insert(point);
            on_face = true;
        } 
        if(point.y > maxy - TOL){
            if(point.x < minx + TOL || point.x > maxx - TOL || point.z < minz + TOL || point.z > maxz - TOL) plus_y_edge.insert(point);
            else plus_y_int.insert(point);
            on_face = true;
        } 
        if(point.z < minz + TOL){
            if(point.x < minx + TOL || point.x > maxx - TOL || point.y < miny + TOL || point.y > maxy - TOL) minus_z_edge.insert(point);
            else minus_z_int.insert(point);
            on_face = true;
        } 
        if(point.z > maxz - TOL){
            if(point.x < minx + TOL || point.x > maxx - TOL || point.y < miny + TOL || point.y > maxy - TOL) plus_z_edge.insert(point);
            else plus_z_int.insert(point);
            on_face = true;
        }

        if(!on_face) interior.push_back(Vec3D(in_points[iter],in_points[iter+1],in_points[iter+2]));
    }

    face_sets.push_back(minus_x_int);
    face_sets.push_back(minus_x_edge);
    face_sets.push_back(plus_x_int);
    face_sets.push_back(plus_x_edge);
    face_sets.push_back(minus_y_int);
    face_sets.push_back(minus_y_edge);
    face_sets.push_back(plus_y_int);
    face_sets.push_back(plus_y_edge);
    face_sets.push_back(minus_z_int);
    face_sets.push_back(minus_z_edge);
    face_sets.push_back(plus_z_int);
    face_sets.push_back(plus_z_edge);
}

//Create a loop of points that defines a face of a prism. Add any points and
//edges to the global manifest for the mesh that are missing
inline void create_face(double mesh_len, double s1, double s2, set<Vec3D> points, Vec3D center, Vec3D v1, Vec3D v2, map<Vec3D, int> &pmap, map<pair<int, int>, int> &emap, vector<int> &tags){

    int iter;
    Vec3D current, next;
    int idx1, idx2, edge_idx;
    double angle, coeff1, coeff2, x, y, z;

    //Make sure face corners are present.
    for(angle = M_PI / 4; angle < 2*M_PI; angle += M_PI / 2){
        coeff1 = s1*cos(angle)/sqrt(2);
        coeff2 = s2*sin(angle)/sqrt(2);
        x = center.x + coeff1*v1.x + coeff2*v2.x;
        y = center.y + coeff1*v1.y + coeff2*v2.y;
        z = center.z + coeff1*v1.z + coeff2*v2.z;
        current = Vec3D(x, y, z);
        if(points.find(current) == points.end()){
            points.insert(current);
        }
    }

    vector<Vec3D> pvec(points.begin(), points.end());

    //Sort points in counter-clockwise order with respect to the outward normal
    //of the face.
    sort(pvec.begin(), pvec.end(), ccw_sort(center, v1, v2));

    for(iter = 0; iter < pvec.size(); iter ++){

        current = pvec[iter];
        next = pvec[(iter + 1) % pvec.size()];

        if(pmap.find(current) == pmap.end()){
            pmap.insert(make_pair(current, pmap.size() + 1));
            gmsh::model::geo::addPoint(current.x, current.y, current.z, mesh_len, pmap.size());
        }
        if(pmap.find(next) == pmap.end()){
            pmap.insert(make_pair(next, pmap.size() + 1));
            gmsh::model::geo::addPoint(next.x, next.y, next.z, mesh_len, pmap.size());
        }

        idx1 = pmap[current];
        idx2 = pmap[next];

        if(emap.find(make_pair(idx1, idx2)) != emap.end()){
            edge_idx = emap[make_pair(idx1, idx2)];
        }
        else if(emap.find(make_pair(idx2, idx1)) != emap.end()){
            edge_idx = -emap[make_pair(idx2, idx1)];
        }
        else{
            emap.insert(make_pair(make_pair(idx1, idx2), emap.size() + 1));
            edge_idx = emap.size();
            gmsh::model::geo::addLine(idx1, idx2, edge_idx);
        }

        tags.push_back(edge_idx);
    }
}

NetworkMeshComplex::~NetworkMeshComplex(){}

void NetworkMeshComplex::tetrahedralize(){}

//Given a set of points to be included in a mesh, network edges, and the maximum
//side length for a mesh cell, find the x, y and z bounds, create an
//axis-aligned rectuangular prism spanning these bounds, enforce inclusion of 
//all specified points, and create lists of output points, index pairs 
//specifying network edges, and point quartets specifying tetrahedra.
void LatticeBasedNMC::tetrahedralize(){

    vector<int> tags;
    int iter, tally, in_idx1, in_idx2, in_idx3, out_idx1, out_idx2, out_idx3;
    double minx, maxx, miny, maxy, minz, maxz;
    map<Vec3D, int> pmap;
    vector<Vec3D> interior;
    vector<set<Vec3D>> face_sets;
    vector<int> minus_x_tags, plus_x_tags, minus_y_tags, plus_y_tags, minus_z_tags, plus_z_tags;
    vector<pair<int, int>> entities;
    map<pair<int, int>, int> emap;
    vector<double> out_points;
    vector<edge> out_edges;
    vector<triple> out_triples;

    minx = bounds.min_x();
    maxx = bounds.max_x();
    miny = bounds.min_y();
    maxy = bounds.max_y();
    miny = bounds.min_z();
    maxz = bounds.max_z();

    //Sort non-corner points into their appropriate categories
    categorize_points(minx, maxx, miny, maxy,  minz, maxz, points, face_sets, interior);

    gmsh::initialize();
    gmsh::model::add("box");

    //Make records of the points and line segments on the cube faces
    //Bottom
    create_face(mesh_len, maxy-miny, maxx-minx, face_sets[9], Vec3D((minx+maxx)/2,(miny+maxy)/2,minz), Vec3D(0,1.,0), Vec3D(1.,0,0), pmap, emap, minus_z_tags);
    //Negative x face
    create_face(mesh_len, maxz-minz, maxy-miny, face_sets[1], Vec3D(minx,(miny+maxy)/2,(minz+maxz)/2), Vec3D(0,0,1.), Vec3D(0,1.,0), pmap, emap, minus_x_tags);
    //Positive x face
    create_face(mesh_len, maxy-miny, maxz-minz, face_sets[3], Vec3D(maxx,(miny+maxy)/2,(minz+maxz)/2), Vec3D(0,1.,0), Vec3D(0,0,1.), pmap, emap, plus_x_tags);
    //Negative y face
    create_face(mesh_len, maxx-minx, maxz-minz, face_sets[5], Vec3D((minx+maxx)/2,miny,(minz+maxz)/2), Vec3D(1.,0,0), Vec3D(0,0,1.), pmap, emap, minus_y_tags);
    //Positive y face
    create_face(mesh_len, maxz-minz, maxx-minx, face_sets[7], Vec3D((minx+maxx)/2,maxy,(minz+maxz)/2), Vec3D(0,0,1.), Vec3D(1.,0,0), pmap, emap, plus_y_tags);
    //Top
    create_face(mesh_len, maxx-minx, maxy-miny, face_sets[11], Vec3D((minx+maxx)/2,(miny+maxy)/2,maxz), Vec3D(1.,0,0), Vec3D(0,1.,0), pmap, emap, plus_z_tags);

    //Add the prism faces
    tally = emap.size();
    gmsh::model::geo::addCurveLoop(minus_z_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 1);
    gmsh::model::geo::addCurveLoop(minus_x_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 2);
    gmsh::model::geo::addCurveLoop(plus_x_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 3);
    gmsh::model::geo::addCurveLoop(minus_y_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 4);
    gmsh::model::geo::addCurveLoop(plus_y_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 5);
    gmsh::model::geo::addCurveLoop(plus_z_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 6);
    gmsh::model::geo::synchronize();

    //Add a record of each point to be inserted into each face
    tally = pmap.size() + 1;
    //Minus x face
    for(Vec3D pt : face_sets[0]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 2);
    tags.clear();

    //Plus x face
    for(Vec3D pt : face_sets[2]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 3);
    tags.clear();

    //Minus y face
    for(Vec3D pt : face_sets[4]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 4);
    tags.clear();

    //Plus y face
    for(Vec3D pt : face_sets[6]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 5);
    tags.clear();

    //Minus z face
    for(Vec3D pt : face_sets[8]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 1);
    tags.clear();

    //Plus z face
    for(Vec3D pt : face_sets[10]){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 2, 6);
    tags.clear();

    //Assemble faces into a boundary representation
    int sl = gmsh::model::geo::addSurfaceLoop({1,2,3,4,5,6});
    int ve = gmsh::model::geo::addVolume({sl});
    gmsh::model::geo::synchronize();
    gmsh::model::addPhysicalGroup(3, {ve}, 1);
    gmsh::model::geo::synchronize();

    //Add a record of each point to be inserted into the interior
    for(Vec3D pt : interior){
        gmsh::model::geo::addPoint(pt.x, pt.y, pt.z, mesh_len, tally);
        tags.push_back(tally ++);
    }
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::embed(0, tags, 3, 1);

    //Construct the mesh
    gmsh::model::mesh::generate(3);

    //Extract points and tetrahedra from the mesh
    gmsh::model::getEntities(entities);
    for(auto e : entities) {
        // Dimension and tag of the entity:
        int dim = e.first, tag = e.second;

        // Get the mesh nodes for the entity (dim, tag):
        vector<size_t> nodeTags;
        vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams, dim, tag);

        if(nodeCoords.size() > 0){
            out_points.insert(out_points.end(), nodeCoords.begin(), nodeCoords.end());
        }

        // Get the mesh elements for the entity (dim, tag):
        vector<int> elemTypes;
        vector<vector<size_t> > elemTags, elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, dim, tag);

        if(dim == 3){
            if(! elemNodeTags.size()) continue;
            tetrahedra.insert(tetrahedra.begin(), elemNodeTags[0].begin(), elemNodeTags[0].end());
        }
    }

    gmsh::finalize();

    //Adjust for one-based indexing in Gmsh
    for(iter = 0; iter < tetrahedra.size(); iter++){
        tetrahedra[iter] -= 1;
    }

    //Reassign edge and triple indices
    tally = 0;
    pmap.clear();
    for(iter = 0; iter < out_points.size(); iter += 3){
        pmap.insert(make_pair(Vec3D(out_points[iter],out_points[iter+1],out_points[iter+2]), tally++));
    }

    for(edge e : edges){
        in_idx1 = get<0>(e);
        out_idx1 = pmap[Vec3D(points[3*in_idx1], points[3*in_idx1+1], points[3*in_idx1+2])];
        in_idx2 = get<1>(e);
        out_idx2 = pmap[Vec3D(points[3*in_idx2], points[3*in_idx2+1], points[3*in_idx2+2])];
        out_edges.push_back(make_tuple(out_idx1, out_idx2));
    }

    for(triple t : triples){
        in_idx1 = get<0>(t);
        out_idx1 = pmap[Vec3D(points[3*in_idx1], points[3*in_idx1+1], points[3*in_idx1+2])];
        in_idx2 = get<1>(t);
        out_idx2 = pmap[Vec3D(points[3*in_idx2], points[3*in_idx2+1], points[3*in_idx2+2])];
        in_idx3 = get<2>(t);
        out_idx3 = pmap[Vec3D(points[3*in_idx3], points[3*in_idx3+1], points[3*in_idx3+2])];
        out_triples.push_back(make_tuple(out_idx1, out_idx2, out_idx3));
    }

    points = out_points;
    edges = out_edges;
    triples = out_triples;
}

//Create a undirected graph in which edges represent pairs of vertices belonging
//to the same tetrahedron and perform a fill-reducing ordering of the associated
//adjacency matrix.
void LatticeBasedNMC::refine_topology(){

    int iter, i, j, result, idx1, idx2, idx3;
    idx_t npts, total;
    set<IdxPair> vertex_pairs;
    map<int, vector<idx_t>> adj_lists;
    idx_t options[METIS_NOPTIONS];
    vector<idx_t> xadj, adjncy, perm, iperm;
    vector<double> permuted_points;
    vector<edge> permuted_edges;
    vector<triple> permuted_triples;

    //Construct adjacency lists
    npts = points.size() / 3;
    for(iter = 0; iter < npts; iter ++){
        adj_lists.insert(make_pair(iter, vector<int>()));
    }

    //If there is a tetrahedralization, enumerate distinct pairs of vertices
    //within the mesh
    if(tetrahedra.size()){
        for(iter = 0; iter < tetrahedra.size(); iter += 4){
            for(i = 0; i < 3; i ++){
                idx1 = iter + i;
                for(j = i + 1; j < 4; j ++){
                    idx2 = iter + j;
                    vertex_pairs.insert(IdxPair(idx1, idx2));
                }
            }
        }
        for(IdxPair vpair : vertex_pairs){
            adj_lists[vpair.idx1].push_back(idx2);
            adj_lists[vpair.idx2].push_back(idx1);
        }
    }
    else if(edges.size()) {
        for(iter = 0; iter < edges.size(); iter ++){
            idx1 = get<0>(edges[iter]);
            idx2 = get<0>(edges[iter]);
            adj_lists[idx1].push_back(idx2);
            adj_lists[idx2].push_back(idx1);
        }
    }

    else {
        cerr << "Neither tetrahedra nor edges are defined.\n";
        return;
    }

    //Build a compressed adjacency list representation for use with METIS
    total = 0;
    xadj.push_back(total);
    for(iter = 0; iter < npts; iter ++){
        total += adj_lists[iter].size();
        xadj.push_back(total);
        adjncy.insert(adjncy.end(), adj_lists[iter].begin(), adj_lists[iter].end());
    }

    //Compute the fill-reducing ordering and apply the permutation to
    //vertices, edges, triples and tetrahedra
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    perm.assign(npts, 0);
    iperm.assign(npts, 0);
    result = METIS_NodeND(&npts, &xadj[0], &adjncy[0], NULL, options, &perm[0], &iperm[0]);
    if(result != METIS_OK){
        cerr << "Reordering of the mesh topology failed.\n";
        exit(result);
    }

    //Vertices
    for(iter = 0; iter < npts; iter ++){
        permuted_points.push_back(3*perm[iter]);
        permuted_points.push_back(3*perm[iter] + 1);
        permuted_points.push_back(3*perm[iter] + 2);
    }
    points = permuted_points;

    //Edges
    if(edges.size()){
        for(iter = 0; iter < edges.size(); iter ++){
            idx1 = get<0>(edges[iter]);
            idx2 = get<1>(edges[iter]);
            permuted_edges.push_back(make_tuple(iperm[idx1], iperm[idx2]));
        }
        edges = permuted_edges;
    }

    //Triples
    if(triples.size()){
        for(iter = 0; iter < triples.size(); iter ++){
            idx1 = get<0>(triples[iter]);
            idx2 = get<1>(triples[iter]);
            idx3 = get<2>(triples[iter]);
            permuted_triples.push_back(make_tuple(iperm[idx1],iperm[idx2],iperm[idx3]));
        }
        triples = permuted_triples;
    }

    //Tetrahedra
    if(tetrahedra.size()){
        for(iter = 0; iter < tetrahedra.size(); iter ++){
            tetrahedra[iter] = iperm[tetrahedra[iter]];
        }
    }
}

inline void create_face_simple(double mesh_len, double s1, double s2, Vec3D center, Vec3D v1, Vec3D v2, map<Vec3D, int> &pmap, map<pair<int, int>, int> &emap, vector<int> &tags){

    int iter, idx1, idx2, edge_idx;
    Vec3D current, next;
    double coeff1, coeff2, x, y, z;
    vector<Vec3D> points;

    //Create the face corners
    for(coeff1 = -.5; coeff1 < 1.5; coeff1++){
        for(coeff2 = -.5; coeff2 < 1.5; coeff2++){
            copy(center, current);
            axpy(v1, current, coeff1*s1);
            axpy(v2, current, coeff2*s2);
            points.push_back(current);
        }
    }

    //Sort points in counter-clockwise order with respect to the outward normal
    //of the face.
    sort(points.begin(), points.end(), ccw_sort(center, v1, v2));

    for(iter = 0; iter < points.size(); iter ++){

        current = points[iter];
        next = points[(iter + 1) % points.size()];

        if(pmap.find(current) == pmap.end()){
            pmap.insert(make_pair(current, pmap.size() + 1));
            gmsh::model::geo::addPoint(current.x, current.y, current.z, mesh_len, pmap.size());
        }
        if(pmap.find(next) == pmap.end()){
            pmap.insert(make_pair(next, pmap.size() + 1));
            gmsh::model::geo::addPoint(next.x, next.y, next.z, mesh_len, pmap.size());
        }

        idx1 = pmap[current];
        idx2 = pmap[next];

        if(emap.find(make_pair(idx1, idx2)) != emap.end()){
            edge_idx = emap[make_pair(idx1, idx2)];
        }
        else if(emap.find(make_pair(idx2, idx1)) != emap.end()){
            edge_idx = -emap[make_pair(idx2, idx1)];
        }
        else{
            emap.insert(make_pair(make_pair(idx1, idx2), emap.size() + 1));
            edge_idx = emap.size();
            gmsh::model::geo::addLine(idx1, idx2, edge_idx);
        }

        tags.push_back(edge_idx);
    }
}

//Tetrahadralize the volume occupied by a network created from a random point
//set. Given the risk of smal, poorly shaped tetrahedra, the vertices of the
//network are not required to be included in the mesh.
void RandomNMCBase::tetrahedralize(){

    vector<int> tags;
    int iter, tally, offset, end;
    double minx, maxx, miny, maxy, minz, maxz;
    map<Vec3D, int> pmap;
    vector<int> minus_x_tags, plus_x_tags, minus_y_tags, plus_y_tags, minus_z_tags, plus_z_tags;
    vector<pair<int, int>> entities;
    map<pair<int, int>, int> emap;
    map<int, vector<int>> point_tet_map;
    shared_ptr<NMBVH> tet_bvh;
    Vec3D query;

    minx = bounds.min_x();
    maxx = bounds.max_x();
    miny = bounds.min_y();
    maxy = bounds.max_y();
    miny = bounds.min_z();
    maxz = bounds.max_z();

    offset = points.size() / 3 - 1;

    gmsh::initialize();
    gmsh::model::add("box");

    //Make records of the points and line segments on the cube faces
    //Bottom
    create_face_simple(mesh_len, maxy-miny, maxx-minx, Vec3D((minx+maxx)/2,(miny+maxy)/2,minz), Vec3D(0,1.,0), Vec3D(1.,0,0), pmap, emap, minus_z_tags);
    //Negative x face
    create_face_simple(mesh_len, maxz-minz, maxy-miny, Vec3D(minx,(miny+maxy)/2,(minz+maxz)/2), Vec3D(0,0,1.), Vec3D(0,1.,0), pmap, emap, minus_x_tags);
    //Positive x face
    create_face_simple(mesh_len, maxy-miny, maxz-minz, Vec3D(maxx,(miny+maxy)/2,(minz+maxz)/2), Vec3D(0,1.,0), Vec3D(0,0,1.), pmap, emap, plus_x_tags);
    //Negative y face
    create_face_simple(mesh_len, maxx-minx, maxz-minz, Vec3D((minx+maxx)/2,miny,(minz+maxz)/2), Vec3D(1.,0,0), Vec3D(0,0,1.), pmap, emap, minus_y_tags);
    //Positive y face
    create_face_simple(mesh_len, maxz-minz, maxx-minx, Vec3D((minx+maxx)/2,maxy,(minz+maxz)/2), Vec3D(0,0,1.), Vec3D(1.,0,0), pmap, emap, plus_y_tags);
    //Top
    create_face_simple(mesh_len, maxx-minx, maxy-miny, Vec3D((minx+maxx)/2,(miny+maxy)/2,maxz), Vec3D(1.,0,0), Vec3D(0,1.,0), pmap, emap, plus_z_tags);

    //Add the prism faces
    tally = emap.size();
    gmsh::model::geo::addCurveLoop(minus_z_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 1);
    gmsh::model::geo::addCurveLoop(minus_x_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 2);
    gmsh::model::geo::addCurveLoop(plus_x_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 3);
    gmsh::model::geo::addCurveLoop(minus_y_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 4);
    gmsh::model::geo::addCurveLoop(plus_y_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 5);
    gmsh::model::geo::addCurveLoop(plus_z_tags, ++tally);
    gmsh::model::geo::addPlaneSurface({tally}, 6);
    gmsh::model::geo::synchronize();

    //Assemble faces into a boundary representation
    int sl = gmsh::model::geo::addSurfaceLoop({1,2,3,4,5,6});
    int ve = gmsh::model::geo::addVolume({sl});
    gmsh::model::geo::synchronize();
    gmsh::model::addPhysicalGroup(3, {ve}, 1);
    gmsh::model::geo::synchronize();

    //Construct the mesh
    gmsh::model::mesh::generate(3);

    //Extract points and tetrahedra from the mesh
    gmsh::model::getEntities(entities);
    for(auto e : entities) {
        // Dimension and tag of the entity:
        int dim = e.first, tag = e.second;

        // Get the mesh nodes for the entity (dim, tag):
        vector<size_t> nodeTags;
        vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams, dim, tag);

        if(nodeCoords.size() > 0){
            points.insert(points.end(), nodeCoords.begin(), nodeCoords.end());
        }

        // Get the mesh elements for the entity (dim, tag):
        vector<int> elemTypes;
        vector<vector<size_t> > elemTags, elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, dim, tag);

        if(dim == 3){
            if(! elemNodeTags.size()) continue;
            tetrahedra.insert(tetrahedra.begin(), elemNodeTags[0].begin(), elemNodeTags[0].end());
        }
    }

    gmsh::finalize();

    for(iter = 0; iter < tetrahedra.size(); iter ++){
        tetrahedra[iter] += offset;
    }

    tet_bvh = make_shared<NMBVH>(points, tetrahedra, bounds);

    for(iter = 0; iter < tetrahedra.size() / 4; iter++){
        point_tet_map.insert(make_pair(iter, vector<int>()));
    }

    for(iter = 0; iter <= offset; iter ++){
        assign(query, points, iter);
        vector<int> hits = tet_bvh->containing_polyhedra(query);
        if(hits.size()){
            point_tet_map[hits[0]].push_back(iter);
        }
        else{
            cerr << "No tetrahedra contained the query point " << query.x << ", " << query.y << ", " << query.z << endl;
        }
    }

    point_in_tet_ranges.clear();
    point_in_tet_records.clear();

    point_in_tet_ranges.push_back(0);
    end = 0;
    for(auto m_iter = point_tet_map.begin(); m_iter != point_tet_map.end(); m_iter ++){
        end += m_iter->second.size();
        point_in_tet_ranges.push_back(end);
        point_in_tet_records.insert(point_in_tet_records.end(), m_iter->second.begin(), m_iter->second.end());
    }
}

//Apply fill-reducing orderings to connectivity graphs for both the edges and
//tetrahedra of the NMC.
void RandomNMCBase::refine_topology(){

    int iter, i, j, idx1, idx2, idx3, result;
    idx_t npts, total;
    set<IdxPair> vertex_pairs;
    map<int, vector<idx_t>> adj_lists;
    idx_t options[METIS_NOPTIONS];
    vector<idx_t> xadj, adjncy, perm, iperm;
    vector<double> permuted_points;
    vector<edge> permuted_edges;
    vector<triple> permuted_triples;

    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    //First refine the topology of the fiber network
    if(edges.size()) {

        npts = num_net_verts;

        //Set up adjacency lists
        for(iter = 0; iter < npts; iter ++){
            adj_lists.insert(make_pair(iter, vector<idx_t>()));
        }
        for(iter = 0; iter < edges.size(); iter += 2){
            idx1 = get<0>(edges[iter]);
            idx2 = get<1>(edges[iter]);
            adj_lists[idx1].push_back(idx2);
            adj_lists[idx2].push_back(idx1);
        }

        //Build a compressed adjacency list representation for use with METIS
        total = 0;
        xadj.push_back(total);
        for(iter = 0; iter < npts; iter ++){
            total += adj_lists[iter].size();
            xadj.push_back(total);
            adjncy.insert(adjncy.end(), adj_lists[iter].begin(), adj_lists[iter].end());
        }

        //Compute the fill-reducing ordering and apply the permutation to
        //vertices, edges, and triples
        perm.assign(npts, 0);
        iperm.assign(npts, 0);
        result = METIS_NodeND(&npts, &xadj[0], &adjncy[0], NULL, options, &perm[0], &iperm[0]);
        if(result != METIS_OK){
            cerr << "Reordering of the mesh topology failed.\n";
            exit(result);
        }

        //Vertices
        for(iter = 0; iter < npts; iter ++){
            permuted_points.push_back(3*perm[iter]);
            permuted_points.push_back(3*perm[iter] + 1);
            permuted_points.push_back(3*perm[iter] + 2);
        }

        //Edges
        for(iter = 0; iter < edges.size(); iter ++){
            idx1 = get<0>(edges[iter]);
            idx2 = get<1>(edges[iter]);
            permuted_edges.push_back(make_tuple(iperm[idx1], iperm[idx2]));
        }
        edges = permuted_edges;

        //Triples
        if(triples.size()){
            for(iter = 0; iter < triples.size(); iter ++){
                idx1 = get<0>(triples[iter]);
                idx2 = get<1>(triples[iter]);
                idx3 = get<2>(triples[iter]);
                permuted_triples.push_back(make_tuple(iperm[idx1],iperm[idx2],iperm[idx3]));
            }
            triples = permuted_triples;
        }
    }

    //Refine the tetrahedral topology
    if(tetrahedra.size()){
        for(iter = 0; iter < tetrahedra.size(); iter += 4){
            for(i = 0; i < 3; i ++){
                idx1 = iter + i - num_net_verts;
                for(j = i + 1; j < 4; j ++){
                    idx2 = iter + j - num_net_verts;
                    vertex_pairs.insert(IdxPair(idx1, idx2));
                }
            }
        }

        adj_lists.clear();
        npts = points.size() / 3 - num_net_verts;
        for(iter = 0; iter < npts; iter ++){
            adj_lists.insert(make_pair(iter, vector<idx_t>()));
        }
        for(IdxPair vpair : vertex_pairs){
            adj_lists[vpair.idx1].push_back(idx2);
            adj_lists[vpair.idx2].push_back(idx1);
        }

        //Build a compressed adjacency list representation for use with METIS
        total = 0;
        xadj.clear();
        xadj.push_back(total);
        for(iter = 0; iter < npts; iter ++){
            total += adj_lists[iter].size();
            xadj.push_back(total);
            adjncy.insert(adjncy.end(), adj_lists[iter].begin(), adj_lists[iter].end());
        }

        //Compute the fill-reducing ordering and apply the permutation to
        //vertices, edges, triples and tetrahedra
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;

        perm.assign(npts, 0);
        iperm.assign(npts, 0);
        result = METIS_NodeND(&npts, &xadj[0], &adjncy[0], NULL, options, &perm[0], &iperm[0]);
        if(result != METIS_OK){
            cerr << "Reordering of the mesh topology failed.\n";
            exit(result);
        }

        //Vertices
        for(iter = 0; iter < npts; iter ++){
            permuted_points.push_back(3*(perm[iter] + num_net_verts));
            permuted_points.push_back(3*(perm[iter] + num_net_verts) + 1);
            permuted_points.push_back(3*(perm[iter] + num_net_verts) + 2);
        }

        //Tetrahedra
        if(tetrahedra.size()){
            for(iter = 0; iter < tetrahedra.size(); iter ++){
                tetrahedra[iter] = iperm[tetrahedra[iter]] + num_net_verts;
            }
        }
    }

    points = permuted_points;
}

//After dilution, determine which points are retained in the network, and which
//triples have retained all members. Construct reduced point and triple lists,
//and relabel elements of edges and triples appropriately.
void NetworkMeshComplex::reassign_edges_triples(){

    map<int, int> pmap;
    int idx1, idx2, idx3;
    vector<double> kept_points;
    vector<triple> kept_triples;

    //Find all retained points and replace the list of points
    for(edge e : edges){
        idx1 = get<0>(e);
        idx2 = get<1>(e);
	    if(pmap.find(idx1) == pmap.end()){
            pmap.insert(make_pair(idx1, pmap.size()));
	        for(int i = 0; i < 3; i++) kept_points.push_back(points[3*idx1+i]);
        }
        if(pmap.find(idx2) == pmap.end()){
            pmap.insert(make_pair(idx2, pmap.size()));
            for(int i = 0; i < 3; i++) kept_points.push_back(points[3*idx2+i]);
        }
    }

    points = kept_points;

    //Reassign edge indices
    for(int iter = 0; iter < edges.size(); iter++){
        idx1 = get<0>(edges[iter]);
        idx2 = get<1>(edges[iter]);
        edges[iter] = make_tuple(pmap[idx1], pmap[idx2]);
    }

    if(! triples.size()) return;

    //Find those triples that have retained all members and replace the list
    //of triples.
    for(triple t : triples){
        idx1 = get<0>(t);
        idx2 = get<1>(t);
        idx3 = get<2>(t);
        if(pmap.find(idx1) != pmap.end()){
            if(pmap.find(idx2) != pmap.end()){
                if(pmap.find(idx3) != pmap.end()){
                    kept_triples.push_back(make_tuple(pmap[idx1], pmap[idx2], pmap[idx3]));
                }
            }
        }
    }

    triples = kept_triples;
}

//Create a lattice-based network from a lat3d file
bool LatticeBasedNMC::build_network(){

    Net3D network_desc;
    LatticeSlice slice;
    LatticeRow row;
    LatticePoint point;
    double curr_x, curr_y, curr_z, neigh_x, neigh_y, neigh_z;
    double true_maxx = -FLT_MAX, true_maxy = -FLT_MAX, true_maxz = -FLT_MAX;
    int slice_iter, row_iter, point_iter, curr_idx, neigh_idx;
    Vec3D curr_point, neighbor;
    map<Vec3D, int> pmap;

    //Make sure all class members are properly initialized
    if(bounds.get_vol() == 0){
        cerr << "The bounding box is empty.\n";
        return false;
    }

    if(lat_file_name == ""){
        cerr << "No lattice file was specified.\n";
        return false;
    }

    if(scale == 0){
        cerr << "The scale was not initialized.\n";
        return false;
    }

    if(! parse_net3d((char *) lat_file_name.c_str(), network_desc)){
        cerr << "Parsing of the network description file failed.\n";
        return false;
    }

    if(scale != 1){
        network_desc.rescale(scale);
    }

    curr_z = bounds.min_z();
    slice_iter = 0;
    while(curr_z <= bounds.max_z()){
        if(curr_z > true_maxz) true_maxz = curr_z;
        curr_y = bounds.min_y();
        row_iter = 0;
        slice = network_desc.slices[slice_iter];
        curr_point.z = curr_z;
        while(curr_y <= bounds.max_y()){
            if(curr_y > true_maxy) true_maxy = curr_y;
            row = slice.rows[row_iter];
            curr_x = bounds.min_x() + row.x_offset;
            point_iter = 0;
            curr_point.y = curr_y;
            while(curr_x <= bounds.max_x()){
                if(curr_x > true_maxx) true_maxx = curr_x;
                point = row.points[point_iter];
                curr_point.x = curr_x;

                if(pmap.find(curr_point) == pmap.end()){
                    pmap.insert(make_pair(curr_point, pmap.size()));
                    points.push_back(curr_x);
                    points.push_back(curr_y);
                    points.push_back(curr_z);
                }
                curr_idx = pmap[curr_point];

                for(Vec3D next_offset : point.offsets){
                    sum(curr_point, next_offset, neighbor);
                    if(bounds.in_bounds(neighbor)){
                        if(pmap.find(neighbor) == pmap.end()){
                            pmap.insert(make_pair(neighbor, pmap.size()));
                            points.push_back(neighbor.x);
                            points.push_back(neighbor.y);
                            points.push_back(neighbor.z);
                        }
                        neigh_idx = pmap[neighbor];
                        edges.push_back(make_tuple(curr_idx, neigh_idx));
                    }
                }
                curr_x += point.to_next;
                point_iter = (point_iter + 1) % row.points.size();
            }
            curr_y += row.y_offset;
            row_iter = (row_iter + 1) % slice.rows.size();
        }
        curr_z += slice.z_offset;
        slice_iter = (slice_iter + 1) % network_desc.slices.size();
    }

    //As the network may not have filled the entire bounding box, trim the
    //bounding box to size.
    bounds.set_max_x(true_maxx);
    bounds.set_max_y(true_maxy);
    bounds.set_max_z(true_maxz);

    //Find all pairs of collinear, contiguous bonds
    compute_triples();

    return true;
}

//Enumerate all triples in a lattice based network. Triples are formed by
//two consectutive, collinear edges belonging to a fiber.
void LatticeBasedNMC::compute_triples(){

    int idx1, idx2, idx3, i, j;
    Vec3D p1, p2, p3, diff1, diff2;
    double len1, len2;
    map<int, vector<int>> neighbor_map;

    for(edge e : edges){
        idx1 = get<0>(e);
        idx2 = get<1>(e);
        if(neighbor_map.find(idx1) == neighbor_map.end()){
            neighbor_map.insert(make_pair(idx1, vector<int>()));
        }
        if(neighbor_map.find(idx2) == neighbor_map.end()){
            neighbor_map.insert(make_pair(idx2, vector<int>()));
        }
        neighbor_map[idx1].push_back(idx2);
        neighbor_map[idx2].push_back(idx1);
    }

    for(auto iter = neighbor_map.begin(); iter != neighbor_map.end(); iter ++){
        idx1 = iter->first;
        assign(p1, points, idx1);
        for(i = 0; i < iter->second.size() - 1; i++){
            idx2 = iter->second[i];
            assign(p2, points, idx2);
	        difference(p2, p1, diff1);
            len1 = magnitude(diff1);

            for(j = i+1; j < iter->second.size(); j++){
                idx3 = iter->second[j];
                assign(p3, points, idx3);
                difference(p3, p1, diff2);
                len2 = magnitude(diff2);

                if(abs(dot(diff1, diff2) + len1*len2) < TOL){
                    triples.push_back(make_tuple(idx2, idx1, idx3));
                    break;
                }
            }
        }
    }

    for(i = 0; i < triples.size(); i++) angles.push_back(M_PI);
}

//Dilute the network, randomly retaining edges with probability p
void NetworkMeshComplex::random_dilute(double p){

    time_t now = time(0);
    mt19937 gen{static_cast<uint32_t>(now)};
	uniform_real_distribution<double> rdist(0.0, 1.0);
    vector<double> reduced_points;
    vector<edge> reduced_edges;
    vector<triple> reduced_triples;
    vector<double> reduced_angles;
    map<int, int> point_map;
    set<IdxPair> idx_pairs;
    int iter, idx1, idx2, idx3;

    //Use a pseudo-random number to decide whether to keep each edge. Make note
    //of which vertices have been retained.
	for(edge e : edges){
        if(p > rdist(gen)){
            idx1 = get<0>(e);
            if(point_map.find(idx1) == point_map.end()){
                point_map.insert(make_pair(idx1, point_map.size()));
                for(iter = 0; iter < 3; iter++){
                    reduced_points.push_back(points[3*idx1+iter]);
                }
            }
            idx2 = get<1>(e);
            if(point_map.find(idx2) == point_map.end()){
                point_map.insert(make_pair(idx2, point_map.size()));
                for(iter = 0; iter < 3; iter++){
                    reduced_points.push_back(points[3*idx2+iter]);
                }
            }
            idx_pairs.insert(IdxPair(idx1, idx2));
            reduced_edges.push_back(make_tuple(point_map[idx1], point_map[idx2]));
        }
    }

    //Determine which triples should be kept, by determining whether both
    //edges involved have been retained.
    for(iter = 0; iter < triples.size(); iter++){
        triple t = triples[iter];
        idx1 = get<0>(t);
        idx2 = get<1>(t);
        idx3 = get<2>(t);
        if(idx_pairs.find(IdxPair(idx1, idx2)) != idx_pairs.end()){
            if(idx_pairs.find(IdxPair(idx2, idx3)) != idx_pairs.end()){
                reduced_triples.push_back(make_tuple(point_map[idx1],point_map[idx2],point_map[idx3]));
                reduced_angles.push_back(angles[iter]);
            }
        }
    }

    points = reduced_points;
	edges = reduced_edges;
    triples = reduced_triples;
    angles = reduced_angles;
}

void NetworkMeshComplex::refine_toplogy(){}

//Break edges in the network into a specified number of sub-segments to
//provide more points at which fibers can flex
void NetworkMeshComplex::add_hinges(int num_segments){

    vector<edge> new_edges;
    int idx1, idx2, prev_idx, curr_idx, iter;
    Vec3D v1, v2, new_vertex, prev_vertex;
    double t;

    for(edge e : edges){
        idx1 = get<0>(e);
        idx2 = get<1>(e);

        assign(v1, points, idx1);
        assign(v2, points, idx2);

        prev_idx = idx1;
        for(iter = 1; iter < num_segments; iter ++){
            t = (double) iter / num_segments;
            curr_idx = points.size() / 3;
            points.push_back(v1.x*(1.-t) + v2.x*t);
            points.push_back(v1.y*(1.-t) + v2.y*t);
            points.push_back(v1.z*(1.-t) + v2.z*t);
            edges.push_back(make_tuple(prev_idx, curr_idx));
            prev_idx = curr_idx;
        }
    }
}

//This function is similar to the one above, but addresses the need to update
//the number of network points, as they are disjoint from the network vertices
void RandomNMCBase::random_dilute(double p){

    time_t now = time(0);
    mt19937 gen{static_cast<uint32_t>(now)};
	uniform_real_distribution<double> rdist(0.0, 1.0);
    vector<double> reduced_points;
    vector<edge> reduced_edges;
    vector<triple> reduced_triples;
    vector<double> reduced_angles;
    map<int, int> point_map;
    set<IdxPair> idx_pairs;
    int iter, idx1, idx2, idx3;

    //Use a pseudo-random number to decide whether to keep each edge. Make note
    //of which vertices have been retained.
	for(edge e : edges){
        if(p > rdist(gen)){
            idx1 = get<0>(e);
            if(point_map.find(idx1) == point_map.end()){
                point_map.insert(make_pair(idx1, point_map.size()));
                for(iter = 0; iter < 3; iter++){
                    reduced_points.push_back(points[3*idx1+iter]);
                }
            }
            idx2 = get<1>(e);
            if(point_map.find(idx2) == point_map.end()){
                point_map.insert(make_pair(idx2, point_map.size()));
                for(iter = 0; iter < 3; iter++){
                    reduced_points.push_back(points[3*idx2+iter]);
                }
            }
            idx_pairs.insert(IdxPair(idx1, idx2));
            reduced_edges.push_back(make_tuple(point_map[idx1], point_map[idx2]));
        }
    }

    //Determine which triples should be kept, by determining whether both
    //edges involved have been retained.
    for(iter = 0; iter < triples.size(); iter++){
        triple t = triples[iter];
        idx1 = get<0>(t);
        idx2 = get<1>(t);
        idx3 = get<2>(t);
        if(idx_pairs.find(IdxPair(idx1, idx2)) != idx_pairs.end()){
            if(idx_pairs.find(IdxPair(idx2, idx3)) != idx_pairs.end()){
                reduced_triples.push_back(make_tuple(point_map[idx1],point_map[idx2],point_map[idx3]));
                reduced_angles.push_back(angles[iter]);
            }
        }
    }

    points = reduced_points;
    num_net_verts = points.size() / 3;
	edges = reduced_edges;
    triples = reduced_triples;
    angles = reduced_angles;
}

//Compute the rest angle of each bending triple
void RandomNMCBase::compute_angles(){

    Vec3D hinge, r1, r2, normal, orth_axis;
    int idx1, idx2, idx3;
    double angle;

    for(triple t : triples){
        idx1 = get<0>(t);
        idx2 = get<1>(t);
        idx3 = get<2>(t);

        assign(hinge, points, idx2);
        assign(r1, points, idx1);
        difference(r1, hinge, r1);
        normalize(r1);
        assign(r2, points, idx3);
        difference(r2, hinge, r2);
        normalize(r2);
        cross(r1, r2, normal);
        normalize(normal);
        cross(normal, r1, orth_axis);
        angle = atan2(dot(r2, orth_axis), dot(r2, r1));
        if(angle < 0) angle += 2*M_PI;
        angles.push_back(angle);
    }
}

//Determine whether, within a shuffling, each successive pair of elements is in
//sorted order.
template <typename T> bool in_order(vector<T> list){
    bool valid = true;
    for(int idx = 0; idx < list.size() - 1; idx += 2){
        valid = list[idx] < list[idx+1];
        if(! valid) break;
    }

    return valid;
}

//Given a range of vertices, find the manner in which to pair edges extending
//from those vertices so that the geometric mean of angles between edge pairs 
//is greatest.
vector<triple> optimal_triples(int hinge, vector<int> end_pts, vector<Vec3D> vertices){

    int i, j;
    map<IdxPair, double> angle_map;
    Vec3D r1, r2, n, e1, e2;
    double angle, product, max_product = 0;
    vector<int> best_shuffling = end_pts;
    vector<triple> triples;

    //Create a map from each possible pairing of triple end points to the
    //angle between the line segments terminating at those endpoints.
    for(i = 0; i < end_pts.size() - 1; i++){
        //Find the displacement from the hinge to the first endpoint
        difference(vertices[end_pts[i]], vertices[hinge], r1);
        for(j = i+1; j < end_pts.size(); j++){
            //Find the displacement from the hinge to the second endpoint
            difference(vertices[end_pts[j]], vertices[hinge], r2);

            //Find the normal to the plane in which the displacement vectors
	        //lie, construct an orthonormal basis, and find the angle between
	        //the displacement vectors.
	        cross(r1, r2, n);
            normalize(n);
            copy(r1, e1);
            normalize(e1);
            cross(n, e1, e2);
            angle = atan2(dot(r2, e2), dot(r2, e1));
            if(angle < 0) angle += 2*M_PI;
            angle_map.insert(make_pair(IdxPair(i, j), angle));
        }
    }

    //Evaluate each valid shuffling of endpoints, and find the one that
    //leads to the set of triples whose hinge angles have the greatest
    //geometric mean.
    do{
        if(in_order(end_pts)){
            product = 1;
            for(i = 0; i < end_pts.size() - 1; i += 2){
                product *= angle_map[IdxPair(end_pts[i], end_pts[i+1])];
            }
            if(product > max_product){
                max_product = product;
                best_shuffling.assign(end_pts.begin(), end_pts.end());
            }
        }
    } while (next_permutation(end_pts.begin(), end_pts.end()));

    for(i = 0; i < best_shuffling.size() - 1; i += 2){
        triples.push_back(make_tuple(best_shuffling[i], hinge, best_shuffling[i+1]));
    }

    return triples;
}

void RandomNMCBase::compute_triples(){

    int idx1, idx2;
    map<int, vector<int>> neighbor_map; 
    vector<Vec3D> vecs;

    for(int iter = 0; iter < points.size(); iter += 3){
        vecs.push_back(Vec3D(points[iter],points[iter+1],points[iter+2]));
    }

    for(edge e : edges){
        idx1 = get<0>(e);
        idx2 = get<1>(e);
        if(neighbor_map.find(idx1) == neighbor_map.end()){
            neighbor_map.insert(make_pair(idx1, vector<int>()));
        }
	
         if(neighbor_map.find(idx2) == neighbor_map.end()){
            neighbor_map.insert(make_pair(idx2, vector<int>()));
        }
        neighbor_map[idx1].push_back(idx2);
        neighbor_map[idx2].push_back(idx1);
    }

    for(auto iter = neighbor_map.begin(); iter != neighbor_map.end(); iter++){
        vector<triple> t_set = optimal_triples(iter->first, iter->second, vecs);
        triples.insert(triples.end(), t_set.begin(), t_set.end());
    }

    compute_angles();
}

//Create a wide set of bounds for creating edges from random point sets.
Bounds3D get_wide_bounds(Bounds3D tight, double padding){

    double minx, maxx, miny, maxy, minz, maxz;

    minx = tight.min_x() - padding;
    maxx = tight.max_x() + padding;
    miny = tight.min_y() - padding;
    maxy = tight.max_y() + padding;
    minz = tight.min_z() - padding;
    maxz = tight.max_z() + padding;

    return Bounds3D(minx, maxx, miny, maxy, minz, maxz);
}

//Attempt to find an intersection between a plane and a line segment
bool segment_plane_intersection(Vec3D normal, Vec3D pt, Vec3D v1, Vec3D v2, Vec3D &intersection){

    Vec3D seg_dir, diff;
    double diff_proj, n_dot_seg_dir, dist;

    difference(v2, v1, seg_dir);
    normalize(seg_dir);
    n_dot_seg_dir = dot(normal, seg_dir);

    //If the line segment is perpendicular to the plane or nearly so, indicate
    //no intersection.
    if(abs(n_dot_seg_dir) < TOL) return false;

    difference(pt, v1, diff);
    dist = dot(normal, diff) / n_dot_seg_dir;
    if(dist > 0 && dist*dist <= distance_sq(v1, v2)){
        copy(v1, intersection);
        intersection.x += dist*seg_dir.x;
        intersection.y += dist*seg_dir.y;
        intersection.z += dist*seg_dir.z;
        return true;
    }

    return false;
}

//Given a line segment with one point inside and one point outside a bounding
//box, find the point where the line segment intersects the bounding box that
//is nearest the interior point.
void closest_intersection(Bounds3D bounds, Vec3D v1, Vec3D &v2){

    Vec3D intersection, closest, plane_pt, normal(-1., 0, 0);
    double dist_sq, min_dist_sq = FLT_MAX;

    //Left face
    plane_pt.x = bounds.min_x();
    plane_pt.y = (bounds.min_y() + bounds.max_y()) / 2.;
    plane_pt.z = (bounds.min_z() + bounds.max_z()) / 2.;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    //Right face
    plane_pt.x = bounds.max_x();
    normal.x = 1;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    //Back face
    plane_pt.x = (bounds.min_x() + bounds.max_x()) / 2.;
    plane_pt.y = bounds.min_y();
    normal.x = 0.;
    normal.y = -1.;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    //Front face
    plane_pt.y = bounds.max_y();
    normal.y = 1.;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    //Bottom face
    plane_pt.y = (bounds.min_y() + bounds.max_y()) / 2.;
    plane_pt.z = bounds.min_z();
    normal.y = 0.;
    normal.z = -1.;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    //Top face
    plane_pt.z = bounds.max_z();
    normal.z = 1.;
    if(segment_plane_intersection(normal, plane_pt, v1, v2, intersection)){
        dist_sq = distance_sq(v1, intersection);
        if(dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            copy(intersection, closest);
        }
    }

    copy(closest, v2);
}

//Given a bounding box and a set of edges, keep edges whose endpoints are
//entirely within the bounding box, trim edges with one point in the bounding
//box, and reject edges with neither point in the bounding box.
void cull_edges(Bounds3D bounds, vector<double> in_pts, vector<edge> in_edges, vector<double> &out_pts, vector<edge> &out_edges){

    bool one_in, two_in;
    Vec3D p1, p2;
    map<Vec3D, int> pmap;
    vector<edge> prelim_edges;
    double length, mean, sdev, min_length;
    int tally = 0, idx1, idx2;

    mean = sdev = 0;

    for(edge e : in_edges){
        assign(p1, in_pts, get<0>(e));
        assign(p2, in_pts, get<1>(e));
        one_in = bounds.in_bounds(p1);
        two_in = bounds.in_bounds(p2);

        if(! one_in && ! two_in) continue;

	    //If either point is not in the box, find where the line segment
	    //intersects the bounding box.
        if(! two_in){
            closest_intersection(bounds, p1, p2);
        }

        else if(! one_in){
            closest_intersection(bounds, p2, p1);
        }

        else{
            length = distance(p1, p2);
            mean += length;
            sdev += length*length;
            tally ++;
        }

        if(pmap.find(p1) == pmap.end()){
            pmap.insert(make_pair(p1, pmap.size()));
            out_pts.push_back(p1.x);
            out_pts.push_back(p1.y);
            out_pts.push_back(p1.z);
        }

        if(pmap.find(p2) == pmap.end()){
            pmap.insert(make_pair(p2, pmap.size()));
            out_pts.push_back(p2.x);
            out_pts.push_back(p2.y);
            out_pts.push_back(p2.z);
        }

        idx1 = pmap[p1];
        idx2 = pmap[p2];

        prelim_edges.push_back(make_tuple(idx1, idx2));
    }

    //Remove especially short edges
    mean /= tally;
    sdev = sqrt(sdev / tally - mean*mean);
    min_length = mean - 2*sdev;

    for(edge next_edge : prelim_edges){
        assign(p1, out_pts, get<0>(next_edge));
        assign(p2, out_pts, get<1>(next_edge));
        length = distance(p1, p2);
        if(length >= min_length) out_edges.push_back(next_edge); 
    }
}

bool RandomPyrochloreNMC::build_network(){

    Bounds3D wide_bounds;
    vector<int> faces;
    vector<double> generators, vertices, midpoints;
    map<Vec3D, int> pmap;
    set<IdxPair> edge_set;
    map<int, vector<int>> valence_map;
    Vec3D p1, p2;
    int nx, ny, nz, vnum, face_idx, v_idx, new_idx, idx1, idx2;
    double len_scale = pow(density, -1./3.), x, y, z;
    voro::voronoicell cell;
    vector<edge> wide_edges;

    //Use the specified point process to obtain generating points.
    wide_bounds = get_wide_bounds(bounds, len_scale);
    switch(proc){
        case point_process::poisson:
            generators = poisson_sphere_packing(wide_bounds, pow(DENSITY_TO_LENGTH/density, 1./3.));
            break;
        case point_process::lloyd:
	        generators = compute_hyperuniform_packing(wide_bounds, density, hu_conv_tol, hu_max_iter);
	        break;
        default:
	        break;
    }

    nx = (int) round(wide_bounds.xrange() / len_scale);
    ny = (int) round(wide_bounds.yrange() / len_scale);
    nz = (int) round(wide_bounds.zrange() / len_scale);

    //Build a Voronoi tessellation and find all edges. Also record edge
    //midpoints and keep track of all edges with a given end point.
    voro::container_periodic con(wide_bounds.max_x(), 0, wide_bounds.max_y(), 0, 0, wide_bounds.max_z(), nx, ny, nz, PTS_PER_BOX);
    for(int iter = 0; iter < generators.size(); iter += 3){
        con.put(iter/3, generators[iter],generators[iter+1],generators[iter+2]);
    }
    voro::c_loop_all_periodic cloop(con);
    if(cloop.start()){
        do if(con.compute_cell(cell, cloop)){
            cloop.pos(x, y, z);
            cell.vertices(x, y, z, vertices);
            cell.face_vertices(faces);
            face_idx = 0;
            do {
                vnum = faces[face_idx];
                for(v_idx = 0; v_idx < vnum; v_idx++){
                    idx1 = faces[face_idx + v_idx+1];
		            assign(p1, vertices, idx1);
		            if(pmap.find(p1) == pmap.end()){
                        new_idx = pmap.size();
                        pmap.insert(make_pair(p1, new_idx));
                        valence_map.insert(make_pair(new_idx, vector<int>()));
                    }
                    idx2 = faces[face_idx + (v_idx+1)%vnum+1];
		            assign(p2, vertices, idx2);
		            if(pmap.find(p2) == pmap.end()){
                        new_idx = pmap.size();
                        pmap.insert(make_pair(p2, new_idx));
                        valence_map.insert(make_pair(new_idx, vector<int>()));
                    }
		            idx1 = pmap[p1];
		            idx2 = pmap[p2];
		            IdxPair ipair = IdxPair(idx1, idx2);
		            if(edge_set.find(ipair) == edge_set.end()){
                        new_idx = edge_set.size();
                        edge_set.insert(ipair);
                        midpoints.push_back((p1.x + p2.x) / 2.);
                        midpoints.push_back((p1.y + p2.y) / 2.);
			            midpoints.push_back((p1.z + p2.z) / 2.);
			            valence_map[idx1].push_back(new_idx);
			            valence_map[idx2].push_back(new_idx);
                    }
                }
                face_idx += vnum + 1;
            } while(face_idx < faces.size());
        } while(cloop.inc());
    }

    //Construct edges by joining midpoints of adjacent Voronoi edges. Discard
    //any edges wholly outside the tighter bounding box, and trim those that
    //intersect it to size.
    for(auto iter = valence_map.begin(); iter != valence_map.end(); iter++){
        for(int i = 0; i < iter->second.size() - 1; i++){
            for(int j = i+1; j < iter->second.size(); j++){
                wide_edges.push_back(make_tuple(iter->second[i],iter->second[j]));
            }
        }
    }

    cull_edges(bounds, midpoints, wide_edges, points, edges);
    //Account for the fact that some edges were rejected for being too short,
    //and some points may need to be ejected and vertex indices may need to be
    //updated.
    reassign_edges_triples();

    compute_triples();

    num_net_verts = points.size() / 3;

    return true;
}
