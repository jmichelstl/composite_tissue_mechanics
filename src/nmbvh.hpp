/*
 * This file defines types for creating bounding volume hierarchies (BVHs)
 * to store polyhedra and support point-in-polyhedron queries. The bounding
 * volumes are axis-aligned bounding boxes, and BVHs are constructed using
 * the linear bounding volume hierarchy techqniue, in which Morton codes are
 * computed from centroids and used to sort geometric objects. From here,
 * a search tree is built in a bottom-up fashion.
 *
 * Code in this library is adapted from Physically Based Rending, Third Edition,
 * by Matt Pharr, Wenzel Jakob and Greg Humphreys, 2018.
 */

#ifndef BVH
#define BVH

#include <vector>
#include <atomic>
#include <algorithm>
#include "geometric_primitives.hpp"
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cmath>
#include <oneapi/tbb.h>
#include <cfloat>

using namespace std;

//Base class for a polyhedron that must support a point-in-polyhedron test
class NMPolyhedron {

    public:

        NMPolyhedron(){}

        virtual ~NMPolyhedron();

        Bounds3D get_bounds(){
            return bounds;
        }

        virtual bool contains(Vec3D point);

    protected:

        int index;
        Bounds3D bounds;
};

//Representation of a tetrahedron in terms of an arbitrary chosen base vertex
//and three vectors from that vertex to the other three vertices
class NMTetrahedron : public NMPolyhedron {

    public:

        NMTetrahedron(vector<double> coordinates, vector<int> tetrahedra, int my_idx);

        bool contains(Vec3D pt);

    private:
        Vec3D v0, d1, d2, d3;
};


//Node type for building a bounding volume hierarchy
struct BVHBuildNode;

//More compact node type for handling queries once the BVH is built
struct BVHSearchNode {

    BVHSearchNode(){}

    Bounds3D bounds;
    union{
        int first_poly;
        int secondChildOffset;
    };
    unsigned int n_polys;
};

//Struct to handle sorting of polyhedra according to their Morton codes
struct MortonInfo;

//Struct to support partitioning Morton encoding-index pairs according to the 
//value of a given bit in a Morton encoding
struct bitcomp;

class NMBVH {

    public:

        //Make a BVH for indexing a set of tetrahedra
        NMBVH(vector<double> coordinates, vector<int> tetrahedra, Bounds3D bounds);

        //Return the indices of any polyhedra in which a point lines
        vector<int> containing_polyhedra(Vec3D query);

    private:

        void index_polyhedra();

        BVHBuildNode *bvh_build(vector<shared_ptr<NMPolyhedron>> &sorted, vector<MortonInfo> &m_info, int start, int end, int bit, int *node_count);

        int flatten_tree(BVHBuildNode *node, int *offset);

        void search_recursive(vector<int> &hits, Vec3D query, int node_idx);

        Bounds3D bounds;
        vector<shared_ptr<NMPolyhedron>> polyhedra;
        vector<BVHSearchNode> s_nodes;
};

#endif
