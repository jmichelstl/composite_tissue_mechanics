/*
 * This code provides a statistical summary of the properties of a network-
 * mesh complex. Four quantities are addressed: edge length, vertex 
 * connectivity, bending angles, and mesh cell volumes. For each quantity,
 * an entry is made in a summary file giving the mean, standard deviation,
 * minimum, and maximum. In addition, a separate histogram file is generated
 * for each quantity. The expected arguments are a network-mesh complex (.nmc)
 * file, and a name pattern for output.
 */

#include "network_mesh_io.hpp"
#include "geometric_primitives.hpp"
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <cmath>
#include <string.h>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;

//Ratio of standard deviation to bin width
#define SDEV_TO_BIN_WIDTH 5.0

#define BIN_TOL 1e-10

//Tags to append to the base name to generate report file names
const char *endings[] = {"_summary.txt", "_length_hist.txt", "_con_hist.txt", "_angle_hist.txt", "_tet_vol_hist.txt"};

//Padding for generating the summary file name
#define SUMMARY_LEN 13

//Padding to add to a character array to accommodate the longest file name
#define MAX_STR_LEN 18 

#define max(a, b) a > b ? a : b

//Given a set of values, compute its min, max, mean and standard deviation,
//and report a set of bins and bin tallies to a histogram file.
void compute_histogram(vector<double> values, double &min, double &max, double &mean, double &sdev, char *filename){

    int bin_tally, bin_num = 0;
    double bin_width, bin_label;
    FILE *report_file;

    report_file = fopen(filename, "w");
    if(! report_file){
        fprintf(stderr, "The file %s could not be opened for writing.\n", report_file);
        return;
    }

    mean = sdev = 0;

    sort(values.begin(), values.end());
    min = *(values.begin());
    max = *(values.rbegin());

    for(double v : values){
        mean += v;
        sdev += v*v;
    }

    mean /= values.size();
    sdev = sqrt(max(sdev / values.size() - mean*mean, 0));
    bin_width = sdev / SDEV_TO_BIN_WIDTH;

    bin_tally = 0;
    for(double v : values){
        if(v < min+(bin_num+1)*bin_width) bin_tally ++;
        else{
            fprintf(report_file, "%1.8lf\t%d\n", min+bin_num*bin_width, bin_tally);
            bin_num++;
            bin_tally = 1;
        }
    }

    fprintf(report_file, "%1.8lf\t%d\n", min+bin_num*bin_width, bin_tally);
    fclose(report_file);
}

//Compute a histogram for integer data
void compute_integer_histogram(vector<int> values, int &min, int &max, double &mean, double &sdev, char *filename){

    int bin_tally, bin_num = 0, bin_width, bin_label;
    FILE *report_file;

    report_file = fopen(filename, "w");
    if(! report_file){
        fprintf(stderr, "The file %s could not be opened for writing.\n", filename);
        return;
    }

    mean = sdev = 0.0;

    sort(values.begin(), values.end());
    min = *(values.begin());
    max = *(values.rbegin());

    for(int v : values){
        mean += v;
        sdev += v*v;
    }

    mean /= values.size();
    sdev = sqrt(max(sdev / values.size() - mean*mean, 0));
    bin_width = (int) ceil(sdev / SDEV_TO_BIN_WIDTH);

    bin_tally = 0;
    for(int v : values){
        if(v < min+(bin_num+1)*bin_width) bin_tally ++;
        else{
            fprintf(report_file, "%d\t%d\n", min+bin_num*bin_width, bin_tally);
            bin_num++;
            bin_tally = 1;
        }
    }

    fprintf(report_file, "%d\t%d\n", min+bin_num*bin_width, bin_tally);

    fclose(report_file);
}

//Determine the connectivity of each network vertex
vector<int> vertex_connectivities(vector<int> edges){

    int iter, idx1, idx2;
    vector<int> connectivities;
    map<int, int> tally_map;

    for(iter = 0; iter < edges.size(); iter += 2){
        idx1 = edges[iter];
        idx2 = edges[iter+1];

        if(tally_map.find(idx1) == tally_map.end()){
            tally_map.insert(make_pair(idx1, 0));
        }

        if(tally_map.find(idx2) == tally_map.end()){
            tally_map.insert(make_pair(idx2, 0));
        }

        tally_map[idx1]++;
        tally_map[idx2]++;
    }

    for(auto t_iter = tally_map.begin(); t_iter != tally_map.end(); t_iter++){
        connectivities.push_back(t_iter->second);
    }

    return connectivities;
}

vector<double> compute_edge_lengths(vector<double> coordinates, vector<int> edges){

    Vec3D p1, p2;
    vector<double> lengths;

    for(int iter = 0; iter < edges.size(); iter += 2){
        assign(p1, coordinates, edges[iter]);
        assign(p2, coordinates, edges[iter+1]);
        lengths.push_back(distance(p1, p2));
    }

    return lengths;
} 

void report_summary_line(FILE *handle, string heading, size_t count, double min, double max, double mean, double sdev){

    fprintf(handle, "%s:\n", heading.c_str());
    fprintf(handle, "\tTotal number: %ld\n", count);
    fprintf(handle, "\tMinimum: %1.8lf\n\tMaximum: %1.8lf\n", min, max);
    fprintf(handle, "\tMean: %1.8lf\n\tStandard deviation: %1.8lf\n",mean,sdev);
}

void report_integer_summary_line(FILE *handle, string heading, size_t count, int min, int max, double mean, double sdev){

    fprintf(handle, "%s:\n", heading.c_str());
    fprintf(handle, "\tTotal number: %ld\n", count);
    fprintf(handle, "\tMinimum: %d\n\tMaximum: %d\n", min, max);
    fprintf(handle, "\tMean: %1.8lf\n\tStandard deviation: %1.8lf\n",mean,sdev);
}

vector<double> calc_volumes(vector<double> points, vector<int> tets){

    Vec3D p1, p2, p3, p4, d1, d2, d3, cross_prod;
    vector<double> volumes;

    for(int iter = 0; iter < tets.size() / 4; iter ++){
        assign(p1, points, tets[iter*4]);
        assign(p2, points, tets[iter*4+1]);
        assign(p3, points, tets[iter*4+2]);
        assign(p4, points, tets[iter*4+3]);

        difference(p2, p1, d1);
        difference(p3, p1, d2);
        difference(p4, p1, d3);

        cross(d2, d3, cross_prod);
    
        volumes.push_back(abs(dot(d1, cross_prod)) / 6);
    }

    return volumes;
}

//Attempt to import a network-mesh complex, and generate information about
//edge lengths, degrees of connectivity, bending angles, and mesh cell volumes.
int main(int argc, char **argv){

    double min, max, mean, sdev;
    int int_min, int_max;
    vector<double> coordinates, angles, lengths, tet_volumes;
    vector<int> edges, triples, tets, vr, cv, connectivities;
    FILE *summary;
    string summary_name, hist_name;
    char c;
    bool have_input_name, have_output_name, disjoint;
    unsigned num_net_verts;

    have_input_name = have_output_name = disjoint = false;

    while((c = getopt(argc, argv, "d:n:o:")) != -1){
        switch(c){
            case 'd':
                if(have_input_name){
                    cerr << "A redundant input file was specified.\n";
                }
                input_name = string(optarg);
                have_input_name = true;
                disjoint = true;
                break;
            case 'n':
                if(have_input_name){
                    cerr << "A redundant input file was specified.\n";
                }
                input_name = string(optarg);
                have_input_name = true;
                break;
            case 'o':
                have_output_name = true;
                break;
            default:
                break;
        }
    }

    //Validate input
    if(! have_input_name || ! have_output_name){
        cerr << "Usage: -n NMC file OR -d DNMC name, -o report name base\n";
        return 2;
    }

    if(disjoint){
        if(extract_disjoint_network_mesh_complex((char *)nmc_name.c_str(), coords, num_net_verts, edges, triples, angles, tets, vr, cv)){
            cerr << "The specified DNMC file could not be parsed.\n";
            return 3;
        }
    }
    else{
        if(! extract_network_mesh_complex(argv[1], coordinates, edges, triples, angles, tets)){
            cerr << "The specified NMC file could not be parsed.\n";
            return 3;
        }
    }

    summary_name = (char *) malloc(sizeof(char)*(strlen(argv[2]) + SUMMARY_LEN));
    hist_name = (char *) malloc(sizeof(char)*(strlen(argv[2]) + MAX_STR_LEN));

    //Make sure the summary can be opened
    sprintf(summary_name, "%s%s", argv[2], endings[0]);
    summary = fopen(summary_name, "w");
    if(! summary){
        fprintf(stderr, "The file %s could not be opened for writing.\n", summary_name);
        return 4;
    }

    //Report statistics about the edge length distribution
    lengths = compute_edge_lengths(coordinates, edges);
    sprintf(hist_name, "%s%s", argv[2], endings[1]);
    compute_histogram(lengths, min, max, mean, sdev, hist_name);
    report_summary_line(summary, "Edge lengths", lengths.size(), min, max, mean, sdev);

    //Compute and report the distribution of vertex connectivities
    connectivities = vertex_connectivities(edges);
    sprintf(hist_name, "%s%s", argv[2], endings[2]);
    compute_integer_histogram(connectivities, int_min, int_max, mean, sdev, hist_name);
    report_integer_summary_line(summary, "Connectivities", connectivities.size(), int_min, int_max, mean, sdev);

    //Report statistics about the angle distribution
    sprintf(hist_name, "%s%s", argv[2], endings[3]);
    compute_histogram(angles, min, max, mean, sdev, hist_name);
    report_summary_line(summary, "Angles", angles.size(), min, max, mean, sdev);

    //Report statistics about the tetrahedron volume distribution
    tet_volumes = calc_volumes(coordinates, tets);
    sprintf(hist_name, "%s%s", argv[2], endings[4]);
    compute_histogram(tet_volumes, min, max, mean, sdev, hist_name);
    report_summary_line(summary, "Tetrahedron volumes", tet_volumes.size(), min, max, mean, sdev);

    fclose(summary);

    return 0;
}
