/*
 * This code provides basic geometric entities to support creation of random
 * fiber networks.
 */

#ifndef GEOMETRICAL_PRIMITIVES
#define GEOMETRICAL_PRIMITIVES

#include <cmath>
#include <vector>

#define TOL 1e-10

//Simple type to store topological relationships
struct IdxPair {

    IdxPair(int arg1, int arg2){
        idx1 = arg1 < arg2 ? arg1 : arg2;
        idx2 = arg1 < arg2 ? arg2 : arg1;
    }

    int idx1, idx2;

    //Utility functions establishing equality and ordering relations
    bool operator == (const IdxPair& pair) const {
        return pair.idx1 == idx1 && pair.idx2 == idx2;
    }

    friend bool operator < (const IdxPair &p1, const IdxPair &p2){
        if(p2.idx2 > p1.idx2) return true;
        else if(p2.idx2 < p1.idx2) return false;
        else if(p2.idx1 > p1.idx1) return true;
        else return false;
    }
};

//Type to represent a three dimensional real-valued vector
struct Vec3D {

    double x, y, z;

    Vec3D(){}

    Vec3D(double my_x, double my_y, double my_z) : x(my_x), y(my_y), z(my_z) {}

    friend bool operator<(const Vec3D &lhs, const Vec3D &rhs){
        if(lhs.x < rhs.x - TOL) return true;
        if(lhs.x > rhs.x + TOL) return false;
        if(lhs.y < rhs.y - TOL) return true;
        if(lhs.y > rhs.y + TOL) return false;
        if(lhs.z < rhs.z - TOL) return true;
        return false;
    }

    friend bool operator==(const Vec3D &lhs, const Vec3D &rhs){
        return !(lhs < rhs || rhs < lhs);
    }
};

//Functions to support vector operations
void sum(Vec3D v1, Vec3D v2, Vec3D &v3);

//Given vectors x and y, and a coefficient coeff, compute y = y + coeff*x
void axpy(Vec3D v1, Vec3D &v2, double coeff);

double magnitude(Vec3D vec);

double magnitude_sq(Vec3D vec);

void difference(Vec3D v1, Vec3D v2, Vec3D &v3);

double dot(Vec3D v1, Vec3D v2);

void cross(Vec3D v1, Vec3D v2, Vec3D &v3);

void normalize(Vec3D &vec);

void scalar_multiply(Vec3D &vec, double coeff);

void copy(Vec3D original, Vec3D &copy);

double distance(Vec3D v1, Vec3D v2);

double distance_sq(Vec3D v1, Vec3D v2);

void assign(Vec3D &vec, double x, double y, double z);

void assign(Vec3D &vec, std::vector<double> vertices, int idx);

void get_midpoint(Vec3D v1, Vec3D v2, Vec3D midpoint);

//Type to define the boundary of a region
class Bounds3D {

    public:

        Bounds3D(){}

        Bounds3D(double minx, double maxx, double miny, double maxy, double minz, double maxz) : minx(minx), maxx(maxx), miny(miny), maxy(maxy), minz(minz), maxz(maxz) {
            vol = (maxx - minx)*(maxy - miny)*(maxz - minz);
        }

        Bounds3D(double args[6]){
            minx = args[0];
            maxx = args[1];
            miny = args[2];
            maxy = args[3];
            minz = args[4];
            maxz = args[5];
            vol = (maxx - minx)*(maxy - miny)*(maxz - minz);
        }

        double xrange(){
            return maxx - minx;
        }

        double yrange(){
            return maxy - miny;
        }

        double zrange(){
            return maxz - minz;
        }

        double max_extent(){
            return fmax(fmax(xrange(),yrange()), zrange());
        }

        bool in_bounds(double x, double y, double z){
            return x>=minx && x<=maxx && y>=miny && y<=maxy && z>=minz&&z<=maxz;
        }

        bool in_bounds(Vec3D p){
            return p.x>=minx && p.x<=maxx && p.y>=miny && p.y<=maxy && p.z>=minz&&p.z<=maxz;
        }

        Vec3D centroid(){
            return Vec3D((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2);
        }

        double get_vol(){
            return vol;
        }

        double min_x(){ return minx;}
        double max_x(){ return maxx;}
        double min_y(){ return miny;}
        double max_y(){ return maxy;}
        double min_z(){ return minz;}
        double max_z(){ return maxz;}

        void set_min_x(double val){minx = val;}
        void set_max_x(double val){maxx = val;}
        void set_min_y(double val){miny = val;}
        void set_max_y(double val){maxy = val;}
        void set_min_z(double val){minz = val;}
        void set_max_z(double val){maxz = val;}

    private:
        double minx, maxx, miny, maxy, minz, maxz, vol;
};

#endif
