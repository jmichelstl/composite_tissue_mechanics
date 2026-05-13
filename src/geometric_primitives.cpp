#include "geometric_primitives.hpp"

//Functions to support vector operations
void sum(Vec3D v1, Vec3D v2, Vec3D &v3){

    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
}

void axpy(Vec3D v1, Vec3D &v2, double coeff){
    v2.x += coeff*v1.x;
    v2.y += coeff*v1.y;
    v2.z += coeff*v1.z;
}

double magnitude(Vec3D vec){

    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

double magnitude_sq(Vec3D vec){

    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

void difference(Vec3D v1, Vec3D v2, Vec3D &v3){

    v3.x = v1.x - v2.x;
    v3.y = v1.y - v2.y;
    v3.z = v1.z - v2.z;
}

double dot(Vec3D v1, Vec3D v2){

    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

void cross(Vec3D v1, Vec3D v2, Vec3D &v3){

    v3.x = v1.y*v2.z - v1.z*v2.y;
    v3.y = v1.z*v2.x - v1.x*v2.z;
    v3.z = v1.x*v2.y - v1.y*v2.x;
}

void normalize(Vec3D &vec){

    double mag = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);

    vec.x /= mag;
    vec.y /= mag;
    vec.z /= mag;
}

void scalar_multiply(Vec3D &vec, double coeff){

    vec.x *= coeff;
    vec.y *= coeff;
    vec.z *= coeff;
}

void copy(Vec3D original, Vec3D &copy){

    copy.x = original.x;
    copy.y = original.y;
    copy.z = original.z;
}

double distance(Vec3D v1, Vec3D v2){

    double dx, dy, dz;

    dx = v1.x - v2.x;
    dy = v1.y - v2.y;
    dz = v1.z - v2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

double distance_sq(Vec3D v1, Vec3D v2){

    double dx, dy, dz;

    dx = v1.x - v2.x;
    dy = v1.y - v2.y;
    dz = v1.z - v2.z;

    return dx*dx + dy*dy + dz*dz;
}

void assign(Vec3D &vec, double x, double y, double z){
    vec.x = x;
    vec.y = y;
    vec.z = z;
}

void assign(Vec3D &vec, std::vector<double> vertices, int idx){
    vec.x = vertices[3*idx];
    vec.y = vertices[3*idx+1];
    vec.z = vertices[3*idx+2];
}

void get_midpoint(Vec3D v1, Vec3D v2, Vec3D midpoint){
    midpoint.x = (v1.x + v2.x) / 2;
    midpoint.y = (v1.y + v2.y) / 2;
    midpoint.z = (v1.z + v2.z) / 2;
}
