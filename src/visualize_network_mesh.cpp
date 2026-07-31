/*
 * This code reads in either an orginal network-mesh complex (.nmc) file alone
 * or a .nmc file with a companion trajectory file pair. When only the first
 * argument is provided, just the original data structure is visualized. When
 * both arguments are provided, a trajectory is visualized, showing the swelling
 * equilipbrium of the NMC and the subsequent relaxed states after uniaxial
 * compression and shear strains have been imposed.
 */

#include <iostream>
#include <vtkCellArray.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkPolyLine.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVertex.h>
#include <vtkTetra.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkXMLMultiBlockDataWriter.h>
#include <vtkCellType.h>
#include <map>
#include "network_mesh_io.hpp"
#include "geometric_primitives.hpp"
#include <vector>
#include <unistd.h>
#include <sstream>

using namespace std;

//Write a PolyLine block to represent the network
void visualize_network(vector<double> coords, vector<int> edges, vtkSmartPointer<vtkPolyData> polydata_ptr){

    map<int, int> pmap;
    vector<Vec3D> point_list;
    int new_idx = 0, vert_idx = 0;

    //Structures for representing points, vertices and lines
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> verts = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkVertex> vertex;
    vtkSmartPointer<vtkPolyLine> line_ptr;

    //Determine which points are actually included in the network, and create
    //a reduced point set
    for(int next_vertex : edges){
        if(next_vertex > coords.size() / 3){
            cerr << "Out of bounds coordinate: " << next_vertex << endl;
        }
        if(pmap.find(next_vertex) == pmap.end()){
            pmap.insert(make_pair(next_vertex, new_idx++));
            point_list.push_back(Vec3D(coords[3*next_vertex],coords[3*next_vertex+1],coords[3*next_vertex+2]));
        }
    }

    for(Vec3D next_point : point_list){
        points->InsertNextPoint(next_point.x, next_point.y, next_point.z);
        vertex = vtkSmartPointer<vtkVertex>::New();
        vertex->GetPointIds()->SetId(0, vert_idx++);
        verts->InsertNextCell(vertex);
    }

    //Construct edges as pairs of indices into the list of retained points
    for(int edge_iter = 0; edge_iter  < edges.size() / 2; edge_iter++){
        line_ptr = vtkSmartPointer<vtkPolyLine>::New();
        line_ptr->GetPointIds()->SetNumberOfIds(2);
        line_ptr->GetPointIds()->SetId(0, pmap[edges[2*edge_iter]]);
        line_ptr->GetPointIds()->SetId(1, pmap[edges[2*edge_iter+1]]);
        lines->InsertNextCell(line_ptr);
    }

    polydata_ptr->SetPoints(points);
    polydata_ptr->SetVerts(verts);
    polydata_ptr->SetLines(lines);
}

//Write a Tetrahedron block to visualize the continuum phase
void visualize_element_mesh(vector<double> coords, vector<int> tets, vtkSmartPointer<vtkUnstructuredGrid> grid_ptr){

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> tet_set = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkTetra> tet_ptr;

    //Load vertex information
    for(int pt_iter = 0; pt_iter < coords.size(); pt_iter += 3){
        points->InsertNextPoint(coords[pt_iter],coords[pt_iter+1],coords[pt_iter+2]);
    }

    //Create a record of each tetrahedron
    for(int tet_iter = 0; tet_iter < tets.size(); tet_iter += 4){
        tet_ptr = vtkSmartPointer<vtkTetra>::New();
        tet_ptr->GetPointIds()->SetId(0, tets[tet_iter]);
        tet_ptr->GetPointIds()->SetId(1, tets[tet_iter+1]);
        tet_ptr->GetPointIds()->SetId(2, tets[tet_iter+2]);
        tet_ptr->GetPointIds()->SetId(3, tets[tet_iter+3]);
        tet_set->InsertNextCell(tet_ptr);
    }

    grid_ptr->SetPoints(points);
    grid_ptr->SetCells(VTK_TETRA, tet_set);
}

//Given coordinate and topology data, create a visualization of a network
//mesh complex.
void visualize_nmc(vector<double> coords, vector<int> edges, vector<int> tets, string output_name){

    vtkSmartPointer<vtkMultiBlockDataSet> mbd = vtkSmartPointer<vtkMultiBlockDataSet>::New();

    //Create the network visualization and add it to a combined structure
    vtkSmartPointer<vtkPolyData> polydata_ptr = vtkSmartPointer<vtkPolyData>::New();
    visualize_network(coords, edges, polydata_ptr);
    mbd->SetBlock(0, polydata_ptr);

    //Create the element mesh visualization and add it to a combined structure
    vtkSmartPointer<vtkUnstructuredGrid> grid_ptr = vtkSmartPointer<vtkUnstructuredGrid>::New();
    visualize_element_mesh(coords, tets, grid_ptr);
    mbd->SetBlock(1, grid_ptr);

    //Write the combined network-mesh structure to a file.
    vtkSmartPointer<vtkXMLMultiBlockDataWriter> writer = vtkSmartPointer<vtkXMLMultiBlockDataWriter>::New();
    writer->SetFileName(output_name.c_str());
    writer->SetInputData(mbd);
    writer->Write();
}

string vtk_name(string base, int seq_num, int dec_digits){

    ostringstream oss;
    int seq_num_digits = get_dec_digits(seq_num);

    oss << base << "_";
    for(int iter = 0; iter < dec_digits - seq_num_digits; iter++){
        oss << "0";
    }

    oss << seq_num << ".vtk";

    return oss.str();
}

//Read either an NMC file alone, or an NMC file paired with a base name for
//output files. In the first case, visualize just the initial network. In
//the second case, visualize the trajectory of the swollen tissue, from its
//initial, swollen state to its relaxed state after a series of compression and
//shear steps.
int main(int argc, char **argv){

    string nmc_name, output_base, xml_name, traj_name;
    bool have_nmc_name, have_output_name , have_traj_name, disjoint;
    vector<double> coords, angles;
    vector<int> edges, trips, tets, vert_ranges, contained_verts; 
    NMCRelaxationRun run;
    FILE *nmt_handle;
    ostringstream oss;
    int dec_digits, seq_num, zlib_status;
    char c;
    unsigned num_net_verts;

    have_nmc_name = have_output_name = have_traj_name = disjoint = false;

    //Look for arguments for the NMC file, trajectory name pattern, and output
    //name pattern
    while((c = getopt(argc, argv, "d:n:t:o:")) != -1){
        switch(c) {
            case 'd':
                if(have_nmc_name){
                    cerr << "A network-mesh file was specified more than once.\n";
                    return 1;
                }
                nmc_name = string(optarg);
                have_nmc_name = true;
                disjoint = true;
                break;
            case 'n':
                if(have_nmc_name){
                    cerr << "A network-mesh file was specified more than once.\n";
                    return 1;
                }
                nmc_name = string(optarg);
                have_nmc_name = true;
                break;
            case 't':
                traj_name = string(optarg);
                have_traj_name = true;
                break;
            case 'o':
                output_base = string(optarg);
                have_output_name = true;
                break;
            case '?' :
                if(optopt == 'n'){
                    cerr << "Option \"n\" requires a .nmc file name.\n";
                    return 1;
                }
                else if(optopt == 't'){
                    cerr << "Option \"t\" requires a trajectory name pattern.\n";
                    return 1;
                }
                else if(optopt == 'o'){
                    cerr << "Option \"o\" requires an output name pattern.\n";
                    return 1;
                }
                else if(isprint(optopt)){
                    fprintf(stderr, "Unrecognized option: %c\n", optopt);
                }
                else{
                    cerr << stderr, "Unknown option character.\n";
                }
                break;
            default:
                break;
        }
    }

    if(! have_nmc_name || ! have_output_name){
        cerr << "At least the .nmc and output names must be specified.\n";
        return 2;
    }

    //Attempt to read the Network Mesh Complex structure
    if(disjoint){
        if(! extract_disjoint_network_mesh_complex((char *)nmc_name.c_str(), coords, num_net_verts, edges, trips, angles, tets, vert_ranges, contained_verts)){
            cerr << "The DNMC file could not be read.\n";
            return 3;
        }
    }
    else{
        if(! extract_network_mesh_complex((char *) nmc_name.c_str(), coords, edges, trips, angles, tets)){
            cerr << "The NMC file could not be read.\n";
            return 3;
        }
    }

    //If specified, attempt to read the description of the relaxation trajectory
    if(have_traj_name){
        if(! import_network_mesh_trajectory(traj_name, run, &nmt_handle)){
            cerr << "The trajectory could not be imported.\n";
            return 4;
        }

        //Report each saved state of the network in turn
        dec_digits = get_dec_digits(run.steps.size());
        seq_num = 0;
        for(RelaxationStep rs : run.steps){
            if((zlib_status = inf(nmt_handle, rs.xsize, &coords[0])) != Z_OK){
                cerr << "Error reading a chunk of coordinate data.\n";
                fclose(nmt_handle);
                return 5;
            }
            visualize_nmc(coords, edges, tets, vtk_name(output_base, seq_num++, dec_digits));
        }

        fclose(nmt_handle);
    }

    //If no trajectory is to be read, just prepare a visualization of the
    //reference sate.
    else{
        oss << output_base << ".vtk";
        visualize_nmc(coords, edges, tets, oss.str());
    }

    return 0;
}
