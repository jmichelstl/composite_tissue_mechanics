#ifndef PARSE_L3D
#define PARSE_L3D

#include <vector>
#include <string>
#include "geometric_primitives.hpp"
#include <iostream>
#include <tinyxml2.h>

using namespace std;
using namespace tinyxml2;

struct LatticePoint {

    double to_next;
    vector<Vec3D> offsets;

    LatticePoint(){}

    void rescale(double scale){
        int iter;

        to_next *= scale;

        for(iter = 0; iter < offsets.size(); iter++){
            offsets[iter].x *= scale;
            offsets[iter].y *= scale;
            offsets[iter].z *= scale;
        }
    }
};

struct LatticeRow {

    double x_offset, y_offset;
    vector<LatticePoint> points;

    LatticeRow(){}

    void rescale(double scale){
        int iter;

        x_offset *= scale;
        y_offset *= scale;

        for(iter = 0; iter < points.size(); iter++){
            points[iter].rescale(scale);
        }
    }
};


struct LatticeSlice {

    double z_offset;
    vector<LatticeRow> rows;

    LatticeSlice(){}

    void rescale(double scale){
        int iter;

        z_offset *= scale;
        for(iter = 0; iter < rows.size(); iter++){
            rows[iter].rescale(scale);
        }
    }
};

struct Net3D {

    vector<LatticeSlice> slices;

    Net3D(){}

    void rescale(double scale){
        int iter;

        for(iter = 0; iter < slices.size(); iter++){
            slices[iter].rescale(scale);
        }
    }
};

//Attempt to read offsets to near neighbors from a lattice point
bool parse_neighbors(XMLElement *point, vector<Vec3D> &neighbors){

    XMLElement *child;
    double dx, dy, dz;

    do{
        child = point->FirstChildElement("neighbor");

        if(child == NULL) continue;

        if(child->QueryDoubleAttribute("dx", &dx) != XML_SUCCESS){
            cerr << "The \"dx\" field could not be found.\n";
            return false;
        }

        if(child->QueryDoubleAttribute("dy", &dy) != XML_SUCCESS){
            cerr << "The \"dy\" field could not be found.\n";
            return false;
        }

        if(child->QueryDoubleAttribute("dz", &dz) != XML_SUCCESS){
            cerr << "The \"dz\" field could not be found.\n";
            return false;
        }

        neighbors.push_back(Vec3D(dx, dy, dz));
        point->DeleteChild(child);
    }while(child != NULL);

    return neighbors.size() > 0;
}

//Attempt to parse the points of a row of a lattice-based network
bool parse_lattice_points(XMLElement *row_ptr, LatticeRow &row){

    XMLElement *child;

    //Try to read the horizontal offset to the next  point
    do{
        child = row_ptr->FirstChildElement("point");

        if(child == NULL) continue;

        LatticePoint next_point;

        if(child->QueryDoubleAttribute("to_next", &(next_point.to_next)) != XML_SUCCESS){
            cerr << "Missing \"to_next\" field\n";
            return false;
        }

        if(! parse_neighbors(child, next_point.offsets)){
            cerr << "Parsing of neighbors failed.\n";
            return false;
        }

        row.points.push_back(next_point);
        row_ptr->DeleteChild(child);
    }while(child != NULL);

    return row.points.size() > 0;
}

//Attempt to parse the rows of a three dimensional, lattice-based network
bool parse_lattice_rows(XMLElement *slice_elem, LatticeSlice &slice){

    XMLElement *child;

    do{
        child = slice_elem->FirstChildElement("row");

        if(child == NULL) continue;

        LatticeRow next_row;

        if(child->QueryDoubleAttribute("x_off", &(next_row.x_offset)) != XML_SUCCESS){
            cerr << "The x offset could not be read.\n";
            return false;
        }

        if(child->QueryDoubleAttribute("y_off", &(next_row.y_offset)) != XML_SUCCESS){
            cerr << "The y offset could not be read.\n";
            return false;
        }

        if(! parse_lattice_points(child, next_row)){
            cerr << "The row points could not be parsed.\n";
            return false;
        }

        slice.rows.push_back(next_row);
        slice_elem->DeleteChild(child);
    }while(child != NULL);

    return slice.rows.size() > 0;
}

//Attempt to parse the slices of a three dimensional, lattice-based network
bool parse_lattice_slices(XMLElement *net3d, Net3D &network){

    XMLElement *child;

    do{
        child = net3d->FirstChildElement("slice");

        if(child == NULL) continue;

        LatticeSlice next_slice;

        if(child->QueryDoubleAttribute("z_off", &(next_slice.z_offset)) != XML_SUCCESS){
            cerr << "The z offset could not be read.\n";
            return false;
        }

        if(! parse_lattice_rows(child, next_slice)){
            cerr << "The row could not be successfully parsed.\n";
            return true;
        }

        network.slices.push_back(next_slice);
        net3d->DeleteChild(child);
    }while(child != NULL);

    return network.slices.size() > 0;
}

//Attempt to parse a top-level net3d element containing the information
//for creating a three dimensional, lattice-based network
bool parse_net3d(char* file_name, Net3D &network_desc){

    XMLDocument doc;
    XMLElement *root;

    if(doc.LoadFile(file_name) != XML_SUCCESS){
        cerr << "The specified file could not be loaded.\n";
        return false;
    }

    root = doc.RootElement();

    if(! root){
        cerr << "The DOM was empty.";
        return false;
    }

    if(strcmp(root->Name(), "net3d") != 0){
        cerr << "The root node must be net3d.\n";
        return false;
    }

    if(! parse_lattice_slices(root, network_desc)){
        cerr << "Parsing of the lattice failed.\n";
        return false;
    }

    return true;
}

#endif
