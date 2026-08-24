#include "nmbvh.hpp"

#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b

extern "C" double orient3d(double *pa, double *pb, double *pc, double *pd);

//Find a union between two bounding boxes
Bounds3D box_union(Bounds3D box1, Bounds3D box2){
    
    double minx, maxx, miny, maxy, minz, maxz;
    
    minx = fmin(box1.min_x(), box2.min_x());
    maxx = fmax(box1.max_x(), box2.max_x());
    miny = fmin(box1.min_y(), box2.min_y());
    maxy = fmax(box1.max_y(), box2.max_y());
    minz = fmin(box1.min_z(), box2.min_z());
    maxz = fmax(box1.max_z(), box2.max_z());

    return Bounds3D(minx, maxx, miny, maxy, minz, maxz);
}

NMPolyhedron::~NMPolyhedron(){}

bool NMPolyhedron::contains(Vec3D pt){
    return false;
}

void add_face(vector<Vec3D> points, int idx1, int idx2, int idx3, Vec3D centroid, int *faces, int face_idx){

    Vec3D diff1, diff2, cross_prod;

    difference(points[idx2], points[idx1], diff1);
    difference(points[idx3], points[idx1], diff2);
    cross(diff1, diff2, cross_prod);

    difference(points[idx1], centroid, diff1);
    if(dot(diff1, cross_prod) < 0){
        faces[3*face_idx] = idx3;
        faces[3*face_idx+2] = idx1;
    }
    else {
        faces[3*face_idx] = idx1;
        faces[3*face_idx+2] = idx3;
    }
    faces[3*face_idx+1] = idx2;
}

NMTetrahedron::NMTetrahedron(vector<double> coordinates, vector<int> tetrahedra, int my_idx){

    double minx, maxx, miny, maxy, minz, maxz;
    vector<Vec3D> points(4, Vec3D());
    Vec3D centroid(0., 0., 0.);

    index = my_idx;
    assign(points[0], coordinates, tetrahedra[4*index]);
    assign(points[1], coordinates, tetrahedra[4*index+1]);
    assign(points[2], coordinates, tetrahedra[4*index+2]);
    assign(points[3], coordinates, tetrahedra[4*index+3]);

    minx = fmin(points[3].x, fmin(points[2].x, fmin(points[1].x, points[0].x)));
    maxx = fmax(points[3].x, fmax(points[2].x, fmax(points[1].x, points[0].x)));
    miny = fmin(points[3].y, fmin(points[2].y, fmin(points[1].y, points[0].y)));
    maxy = fmax(points[3].y, fmax(points[2].y, fmax(points[1].y, points[0].y)));
    minz = fmin(points[3].z, fmin(points[2].z, fmin(points[1].z, points[0].z)));
    maxz = fmax(points[3].z, fmax(points[2].z, fmax(points[1].z, points[0].z)));
    bounds = Bounds3D(minx-TOL, maxx+TOL, miny-TOL, maxy+TOL, minz-TOL, maxz+TOL);

    for(int iter = 0; iter < 4; iter ++){
        vertices[3*iter] = points[iter].x;
        centroid.x += points[iter].x;
        vertices[3*iter + 1] = points[iter].y;
        centroid.y += points[iter].y;
        vertices[3*iter + 2] = points[iter].z;
        centroid.z += points[iter].z;
    }
    centroid.x /= 4.;
    centroid.y /= 4.;
    centroid.z /= 4.;

    //Make sure each face is properly oriented
    add_face(points, 0, 1, 2, centroid, faces, 0);
    add_face(points, 0, 1, 3, centroid, faces, 1);
    add_face(points, 0, 2, 3, centroid, faces, 2);
    add_face(points, 1, 2, 3, centroid, faces, 3);
}

bool NMTetrahedron::contains(Vec3D pt){

    double pd[3];
    int idx1, idx2, idx3;

    pd[0] = pt.x;
    pd[1] = pt.y;
    pd[2] = pt.z;

    for(int iter = 0; iter < 4; iter++){
        idx1 = faces[3*iter];
        idx2 = faces[3*iter+1];
        idx3 = faces[3*iter+2];
        if(orient3d(vertices+3*idx1, vertices+3*idx2, vertices+3*idx3, pd) < -TOL){
            return false;
        }
    }

    return true; 
}

//Node type for building a bounding volume hierarchy
struct BVHBuildNode {

    void init_internal(BVHBuildNode *left_child, BVHBuildNode *right_child){
        n_polys = 0;
        children[0] = left_child;
        children[1] = right_child;
        bounds = box_union(left_child->bounds, right_child->bounds);
    }

    void init_leaf(int my_first_poly, int my_n_polys, const Bounds3D &b) {
        first_poly = my_first_poly;
        n_polys = my_n_polys;
        children[0] = nullptr;
        children[1] = nullptr;
        bounds = b;
    }

    Bounds3D bounds;
    BVHBuildNode *children[2];
    int first_poly, n_polys;
};

//Struct to handle sorting of polyhedra according to their Morton codes
struct MortonInfo {

    MortonInfo(){}

    int poly_idx;
    uint32_t morton_code;
};

//Struct to support partitioning Morton encoding-index pairs according to the 
//value of a given bit in a Morton encoding
struct bitcomp {

    bitcomp(uint32_t mask) : mask(mask) {}

    inline bool operator()(const MortonInfo &arg) {
        return arg.morton_code & mask;
    }

    uint32_t mask;
};

//Compress a coordinate value to the interval [0,1), then map this value to
//an integer in the range [0,1023]
inline uint32_t map_to_interval(double x, double xmin, double range){

    return (uint32_t) ((x - xmin) / range * 1024);
}

//Left-shift bits of a number to place two zero bits between each original
//bit of a number.
inline uint32_t left_shift_3(uint32_t x) {
    if (x == (1 << 10)) --x;

    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;

    return x;
}

//Produce a Morton encoding of a point
inline uint32_t encode_morton(Vec3D pt, double xmin, double xrange, double ymin, double yrange, double zmin, double zrange){

    uint32_t xcode, ycode, zcode;

    xcode = left_shift_3(map_to_interval(pt.x, xmin, xrange));
    ycode = left_shift_3(map_to_interval(pt.y, ymin, yrange));
    zcode = left_shift_3(map_to_interval(pt.z, zmin, zrange));

    return xcode | (ycode << 1) | (zcode << 2);
}

//Partition a sorted list of polyhedra recursively to build a tree
BVHBuildNode * NMBVH::bvh_build(vector<shared_ptr<NMPolyhedron>> &sorted, vector<MortonInfo> &m_info, int start, int end, int bit, int *node_count){

    BVHBuildNode *node = (BVHBuildNode *) malloc(sizeof(BVHBuildNode));
    *node_count = *node_count + 1;

    //If the range between the start and end indices is 1, make a leaf 
    if(end - start == 1){
        node->init_leaf(sorted.size(), 1, polyhedra[m_info[start].poly_idx]->get_bounds());
        sorted.push_back(polyhedra[m_info[start].poly_idx]);
    }

    //Try to split the current subset of polyhedra along some axis. If
    //splitting is not achieved, place all nodes in a leaf.
    else{
        auto iter1 = next(m_info.begin(), start);
        auto iter2 = next(m_info.begin(), end);
        do{
            auto split = partition(iter1, iter2, bitcomp(1 << (bit-1)));
            if( split != iter1 && split != iter2){
                node->init_internal(bvh_build(sorted, m_info, start, start+distance(iter1, split), bit-1, node_count), bvh_build(sorted, m_info, start+distance(iter1,split), end, bit-1, node_count));
                return node;
            }

            bit --;
        } while(bit > 0);

        //If the loop terminates, no split was achieved, and all polyhedra
        //should be placed in a new leaf
        Bounds3D leaf_bounds = polyhedra[m_info[start].poly_idx]->get_bounds();
        sorted.push_back(polyhedra[m_info[start].poly_idx]);
        for(auto iter = next(iter1); iter != iter2; iter ++){
            leaf_bounds = box_union(leaf_bounds, polyhedra[(*iter).poly_idx]->get_bounds());
            sorted.push_back(polyhedra[(*iter).poly_idx]);
        }
        node->init_leaf(sorted.size()+start-end, end - start, leaf_bounds);
    }

    return node;
}

int NMBVH::flatten_tree(BVHBuildNode *node, int *offset){

    BVHSearchNode *s_node = &s_nodes[*offset];
    s_node->bounds = node->bounds;
    int this_offset = (*offset)++;

    if(node->n_polys > 0){
        s_node->n_polys = node->n_polys;
        s_node->first_poly = node->first_poly;
    }

    else{
        s_node->n_polys = 0;
        flatten_tree(node->children[0], offset);
        s_node->secondChildOffset = flatten_tree(node->children[1], offset);
    }

    free(node);
    return this_offset;
}

void NMBVH::index_polyhedra() {

    int node_count = 0, offset = 0, iter;
    vector<shared_ptr<NMPolyhedron>> sorted;
    vector<MortonInfo> m_info(polyhedra.size(), MortonInfo());

    //Compute Morton encodings
    tbb::parallel_for(tbb::blocked_range<size_t>(0, polyhedra.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                m_info[index].poly_idx = index;
                m_info[index].morton_code = encode_morton(polyhedra[index]->get_bounds().centroid(), bounds.min_x(), bounds.xrange(), bounds.min_y(), bounds.yrange(), bounds.min_z(), bounds.zrange());
            }
        }
    );

    //Create the initial search tree with build nodes
    BVHBuildNode *root = bvh_build(sorted, m_info, 0, m_info.size(), 30, &node_count);
    polyhedra = sorted;

    //Create the more efficient tree representation with search nodes
    for(int iter = 0; iter < node_count; iter++){
        s_nodes.push_back(BVHSearchNode());
    }
    flatten_tree(root, &offset);
}

NMBVH::NMBVH(vector<double> coordinates, vector<int> tetrahedra, Bounds3D bounds) : bounds(bounds) {

    tbb::concurrent_vector<shared_ptr<NMTetrahedron>> nm_tetrahedra;

    //Initialize tetrahedra and add them to the master list of polyhedra
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tetrahedra.size() / 4),
        [&](const tbb::blocked_range<size_t>& r) {
            for(size_t index = r.begin(); index != r.end(); index++){
                nm_tetrahedra.push_back(make_shared<NMTetrahedron>(coordinates, tetrahedra, index));
            }
        }
    );
    polyhedra.assign(nm_tetrahedra.begin(), nm_tetrahedra.end());
    index_polyhedra();
}

void NMBVH::search_recursive(vector<int> &hits, Vec3D query, int node_idx){

    Bounds3D bounds = s_nodes[node_idx].bounds;
    if(s_nodes[node_idx].n_polys > 0){
        int poly_start = s_nodes[node_idx].first_poly;
        int poly_end = poly_start + s_nodes[node_idx].n_polys;
        for(int poly_idx = poly_start; poly_idx < poly_end; poly_idx++){
            if(polyhedra[poly_idx]->contains(query)){
                hits.push_back(polyhedra[poly_idx]->get_index());
            }
        }
    }

    else{
        if(s_nodes[node_idx+1].bounds.in_bounds(query)){
            search_recursive(hits, query, node_idx+1);
        }
        int offset = s_nodes[node_idx].secondChildOffset;
        if(s_nodes[offset].bounds.in_bounds(query)){
            search_recursive(hits, query, offset);
        }
    }
}

vector<int> NMBVH::containing_polyhedra(Vec3D query){

    vector<int> hits;

    if(s_nodes.size() == 0) return hits;

    search_recursive(hits, query, 0);

    return hits;
}
