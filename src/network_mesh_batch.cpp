/*
 * This code reads a batch file, extracts instructions for building either
 * lattice-based or off-latice networks and accompanying tetrahedral finite
 * element meshes, and writes the resuilting data structures to files.
 */

#include "network_mesh_complex.hpp"
#include "network_mesh_io.hpp"
#include <string>
#include <string.h>
#include <memory>
#include <fstream>
#include <cmath>
#include <sstream>
#include <cfloat>

using namespace std;

extern "C" void exactinit();

//Types to handle different classes of network construction protocol

//Different types of random networks
//For now, only the pyrochlore lattice is supported. This may change in the
//future.
enum class point_decorations {pyrochlore};

//Different types of construction protocol
enum class construction_type {lattice, off};

//Different states of the batch file parser
enum class nmc_parser_state {bounds, construction, bond_prob, elem_vol, output, 
	complete};

//Generic virtual type for construction parameters
class ConstructParams {

    public:

        ConstructParams(){}

        virtual ~ConstructParams(){}
};

//Type for lattice-based network construction
class LatticeBasedParams : public ConstructParams {

    public:

        LatticeBasedParams(string my_file, double my_scale){
            n3d_file = my_file;
	        scale = my_scale;
        }

	    string get_file(){
            return n3d_file;
        }

	    double get_scale(){
            return scale;
        }

    private:
        string n3d_file;
        double scale;
};

//Type for off-lattice network construction
class OffLatticeParams : public ConstructParams {

    public:

        OffLatticeParams(point_process my_pt_proc, point_decorations my_dec, double my_dens){
            proc = my_pt_proc;
            dec = my_dec;
	        density = my_dens;
        }

        point_process get_process(){
            return proc;
        }

        point_decorations get_decoration(){
            return dec;
        }

        double get_density(){
            return density;
        }

    private:
	    point_process proc;
	    point_decorations dec;
	    double density;

};

//Information about bond occupation probabilities
struct BOPParams {

    BOPParams(){}

    BOPParams(double my_min){
        min = my_min;
        max = my_min;
        inc = my_min;
        dec_digits = 0;
    }

    BOPParams(double my_min, double my_max, double my_inc, int my_digits){
        min = my_min;
        max = my_max;
        inc = my_inc;
        dec_digits = my_digits;
    }

    //Minimum and maximum values and increment for bond occupation probabilities
    double min, max, inc;
    //Number of decimal digits used in appending information to file names about
    //the bond occupation probability
    int dec_digits;
};

//Type to organize a set of instructions for a batch of network-mesh complexes
struct NMCJob {

    NMCJob(){}

    double max_vol;
    Bounds3D bounds;
    construction_type ctype;
    shared_ptr<ConstructParams> constr_par;
    BOPParams bop_par;
    int realizations;
    string base_name;
};

//Parse bounds
bool parse_bounds(string line, double bounds[6]){

    if(sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf", &bounds[0], &bounds[1], &bounds[2], &bounds[3], &bounds[4], &bounds[5]) < 6){
        cerr << "Six numbers could not be read in the bounds field.\n";
        return false;
    }

    if(bounds[1] < bounds[0] || bounds[3] < bounds[2] || bounds[5] < bounds[4]){
        cerr << "The bounds were improperly ordered.\n";
        return false;
    }

    return true;
}

vector<string> split(string input, char delim){
    vector<string> result;
    size_t start, iter, len;
    bool reading_token = false;

    for(iter = 0; iter < input.size(); iter++){
        if(delim != input[iter]){
            if(!reading_token){
                reading_token = true;
                start = iter;
            }
        }

        else{
            if(reading_token){
                reading_token = false;
                result.push_back(input.substr(start, iter - start));
            }
        }
    }

    if(reading_token) result.push_back(input.substr(start, iter - start));

    return result;
}

//Parse the construction protocol
bool parse_construction(string line, construction_type &ct, shared_ptr<ConstructParams> &con_ptr){

    point_process pt_proc;
    point_decorations dec;
    int num_read;
    double scale, density;

    vector<string> tokens = split(line, ' ');
    if(! tokens.size()){
        cerr << "No arguments were read.\n";
        return false;
    }

    //Lattice based
    if(tokens[0] == "lattice"){

        if(tokens.size() < 3){
            cerr << "Too few construction parameters were specified.\n";
            return false;
        }

        if(!(num_read=sscanf(tokens[2].c_str(), "%lf", &scale)) || scale <= 0){
            cerr << "The scale factor must be a positive number.\n";
            return false;
        }

        con_ptr = make_shared<LatticeBasedParams>(tokens[1], scale);
        ct = construction_type::lattice;
        return true;
    }

    //Off-lattice
    else if(tokens[0] == "off"){

        if(tokens.size() < 4){
            cerr << "Too few construction parameters were specified.\n";
            return false;
        }

        if(tokens[1] == "poisson") pt_proc = point_process::poisson;
        else if(tokens[1] == "lloyd") pt_proc = point_process::lloyd;
        else{
            cerr << "Unrecognized point process\n";
            return false;
        }

        if(tokens[2] == "pyrochlore"){
            dec = point_decorations::pyrochlore;
        }
        else{
            cerr << "Unrecognized point decoration\n";
            return false;
        }

        if(! (num_read=sscanf(tokens[3].c_str(), "%lf", &density)) || density <= 0){
            cerr << "The density must be a positive number.\n";
            return false;
        }

        con_ptr = make_shared<OffLatticeParams>(pt_proc, dec, density);
        ct = construction_type::off;
        return true;
    }

    //Unsupported
    else{
        cerr << "Unsupported protocol\n";
        return false;
    }
}

bool in_bounds(double value, double min, double max){
    return value >= min && value <= max;
}

//Obtain bond occupation probabilities
bool parse_bond_occ_prob(string line, BOPParams &bpar){

    double min, max, inc;
    int num_read, dec_digits;

    num_read = sscanf(line.c_str(), "%lf %lf %lf %d", &min, &max, &inc, &dec_digits);

    if(num_read == 1){
        if(in_bounds(min, FLT_MIN, 1)){ 
            bpar = BOPParams(min);
            return true;
        }

        else{
            cerr << "The minimum bond occupation probability must be between zero and one.\n";
            return false;
        }
    }

    else if(num_read == 4){
        if( ! in_bounds(min, FLT_MIN, 1)){
            cerr << "The minimum bond occupation probability must be between zero and one.\n";
            return false;
        }
        if( ! in_bounds(max, FLT_MIN, 1)){
            cerr << "The maximum bond occupation probability must be between zero and one.\n";
            return false;
        }
        if( ! in_bounds(inc, FLT_MIN, FLT_MAX)){
            cerr << "The increment must be greater than zero.\n";
            return false;
        }
        if(dec_digits <= 0){
            cerr << "The number of digits for reporting the bond occupation probability must be non-negative.\n";
            return false;
        }
        bpar = BOPParams(min, max, inc, dec_digits);
        return true;
    }

    else{
        cerr << "There should be either one or four parameters specified for the bond occupation probability field.\n";
        cerr << "Actual number: " << num_read << endl;
        return false;
    }
}

//Obtain the number of realizations for each bond occupation probability and
//the base name for reporting
bool get_output_params(string line, int &realizations, string &base_name){

    int num_read;

    vector<string> tokens = split(line, ' ');

    if(tokens.size() < 2){
        cerr << "Expected a number of realizations and base name.\n";
        cerr << "Number of arguments: " << tokens.size() << endl;
        return false;
    }

    if((num_read = sscanf(tokens[0].c_str(), "%d", &realizations)) == 0){
        cerr << "The number of realizations could not be read.\n";
        return false;
    }

    else if(realizations <= 0){
        cerr << "The number of realizations must be positive.\n";
        return false;
    }

    base_name = tokens[1];

    return true;
}

//Attempt to parse a batch file and create a set of instructions from it
bool parse_nmc_batch_file(string file_name, NMCJob &job){

    nmc_parser_state state = nmc_parser_state::bounds;
    ifstream batch_file;
    string nextline, base_name;
    double max_vol, bounds[6];
    construction_type ct;
    shared_ptr<ConstructParams> con_ptr;
    BOPParams bpar;
    int num_read, realizations;

    batch_file.open(file_name);

    if(! batch_file.is_open() || batch_file.eof()){
        cerr << "No information could be read from the specified file.\n";
        return false;
    }

    //Attempt to read each batch file field in turn, ignoring empty lines and
    //comments set off by a pound sign
    do{

        getline(batch_file, nextline);

        if(nextline == "" || nextline[0] == '#') continue;

        switch(state){
            case nmc_parser_state::bounds :
                if(! parse_bounds(nextline, bounds)){
                    cerr << "Parsing of bounds failed.\n";
                    batch_file.close();
                    return false;
                }
                state = nmc_parser_state::construction;
                break;
            case nmc_parser_state::construction :
                if(! parse_construction(nextline, ct, con_ptr)){
                    cerr << "Parsing of construction protocol failed.\n";
                    batch_file.close();
                    return false;
                }
                state = nmc_parser_state::bond_prob;
                break;
            case nmc_parser_state::bond_prob :
                if(! parse_bond_occ_prob(nextline, bpar)){
                    cerr << "Parsing of bond occupation probability failed.\n";
                    batch_file.close();
                    return false;
                }
                state = nmc_parser_state::elem_vol;
                break;
            case nmc_parser_state::elem_vol :
                if((num_read = sscanf(nextline.c_str(), "%lf", &max_vol)) == 0){
                    cerr << "Expected a number for the maximum volume.\n";
                    batch_file.close();
                    return false;
                }
                else if(max_vol < 0){
                    cerr << "The maximum volume must be non-negative.\n";
                    batch_file.close();
                    return false;
                }
                state = nmc_parser_state::output;
                break;
            case nmc_parser_state::output :
                if(! get_output_params(nextline, realizations, base_name)){
                    cerr << "Parsing of output parameters failed.\n";
                    batch_file.close();
                    return false;
                }
                state = nmc_parser_state::complete;
                break;
            default:
                break;
        }

    } while(! batch_file.eof() && state != nmc_parser_state::complete);

    batch_file.close();

    //If the parser state is not complete, not all data were obtained
    if(state != nmc_parser_state::complete){
        cerr << "Not all fields could be read from the batch file.\n";
        return false;
    }

    //Upon success, load all construction parameters and report success
    job.bounds = Bounds3D(bounds);
    job.ctype = ct;
    job.constr_par = con_ptr;
    job.bop_par = bpar;
    job.max_vol = max_vol;
    job.realizations = realizations;
    job.base_name = base_name;

    return true;
}

//Generate an output file name when nothing needs to be appended to the base 
//name but a file extension
string output_name_simple(string base, string ext){

    ostringstream oss;

    oss << base << ext;

    return oss.str();
}

//Given a base name, a number of decimal digits for reporting a bond occupation
//probability and a sequence number, and the accompanying BOP and sequence #,
//construct an output file name.
string output_name_verbose(string base, int bop_digits, int total_seq_digits, double bop, int seq_num, string ext){

    ostringstream oss;
    int curr_seq_digits, bop_rep, bop_rep_digits, iter;

    oss << base << "_";

    bop_rep = (int) round(bop * pow(10, bop_digits - 1));
    bop_rep_digits = get_dec_digits(bop_rep);
    curr_seq_digits = get_dec_digits(seq_num);

    for(iter = 0; iter < bop_digits - bop_rep_digits; iter ++){
        oss << "0";
    }
    oss << bop_rep << "_";

    for(iter = 0; iter < total_seq_digits - curr_seq_digits; iter ++){
        oss << "0";
    }
    oss << seq_num << ext;

    return oss.str();
}

//Slightly different procedures are used for lattice based and off-lattice
//models. Lattice-based batch creation can benefit from the fact that, for
//every realization, the sets of all allowable vertices, edges, and triples
//are always the same.

void lattice_based_batch(NMCJob job, bool use_verbose, int real_dec_digits){

    double bop;
    string output_name, ext = ".nmc";

    //Create a fully connected network. Copies of this network can be diluted
    //as needed.
    shared_ptr<LatticeBasedParams> lbp = static_pointer_cast<LatticeBasedParams>(job.constr_par);
    shared_ptr<LatticeBasedNMC> lbnmc = make_shared<LatticeBasedNMC>(job.bounds, lbp->get_file(), lbp->get_scale(), job.max_vol);
    lbnmc->build_network();

    for(bop = job.bop_par.min; bop <= job.bop_par.max + TOL; bop += job.bop_par.inc){
        for(int iter = 1; iter <= job.realizations; iter++){

            //Generate the output name and report the structure to a file
            if(use_verbose){
                output_name = output_name_verbose(job.base_name, job.bop_par.dec_digits, real_dec_digits, bop, iter, ext);
            }
            else output_name = output_name_simple(job.base_name, ext);

            //Build and dilute a network copy and tetrahedralize its convex hull
            shared_ptr<LatticeBasedNMC> lbnmc_copy = make_shared<LatticeBasedNMC>(*lbnmc);
            if(bop < 1){
                lbnmc_copy->random_dilute(bop);
            }
            if(job.max_vol > 0) lbnmc_copy->tetrahedralize();
            report_network_mesh_complex((char *) output_name.c_str(), lbnmc_copy);
        }
    }
}

void off_lattice_batch(NMCJob job, bool use_verbose, int real_dec_digits){

    double bop;
    string output_name, ext;
    shared_ptr<OffLatticeParams> olp = static_pointer_cast<OffLatticeParams>(job.constr_par);

    if(job.max_vol > 0) ext = ".dnmc";
    else ext = ".nmc";

    for(bop = job.bop_par.min; bop <= job.bop_par.max + TOL; bop += job.bop_par.inc){
        for(int iter = 1; iter <= job.realizations; iter++){

            //Generate the output name and report the structure to a file
            if(use_verbose){
                output_name = output_name_verbose(job.base_name, job.bop_par.dec_digits, real_dec_digits, bop, iter, ext);
            }
            else output_name = output_name_simple(job.base_name, ext);

            //Build the network and tetrahedralize its convex hull
            
            //TODO: If more decorations are supported in the future, more cases
            //will need to be added here.
            if(olp->get_decoration() == point_decorations::pyrochlore){  
                shared_ptr<RandomPyrochloreNMC> rpnmc = make_shared<RandomPyrochloreNMC>(job.bounds, olp->get_density(), olp->get_process(), job.max_vol);
                rpnmc->build_network();
                if(bop < 1){
                    rpnmc->random_dilute(bop);
                }

                if(job.max_vol > 0){
                    rpnmc->tetrahedralize();
                    report_disjoint_network_mesh_complex((char *) output_name.c_str(), rpnmc);
                }
                else{
                    report_network_mesh_complex((char *) output_name.c_str(), rpnmc);
                }
            }
        }
    }
}

//Try to extract instructions from a batch file, build a series of NMCs, and
//write them to disk.
int main(int argc, char **argv){

    NMCJob job;
    double min_bp, max_bp, bp_inc;
    int iter, real_dec_digits;
    double bop;
    bool use_verbose;
    string output_name;

    if(argc < 2){
        cerr << "Usage: NMC batch file\n";
        return 1;
    }

    if(! parse_nmc_batch_file(string(argv[1]), job)){
        cerr << "The batch file could not be successfully parsed.\n";
        return 2;
    }

    if(job.realizations > 1) real_dec_digits = get_dec_digits(job.realizations);
    else real_dec_digits = 1;

    use_verbose = job.realizations > 1 || job.bop_par.min != job.bop_par.max;

    if(job.ctype == construction_type::lattice){
        lattice_based_batch( job, use_verbose, real_dec_digits);
    }

    else{
        exactinit();
        off_lattice_batch(job, use_verbose, real_dec_digits);
    }

    return 0;
}
