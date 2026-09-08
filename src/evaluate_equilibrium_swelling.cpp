/*
 * This code is intended to check the equilibrium swelling implied by a gel
 * constitutive model.
 */
#include <iostream>
#include <vector>
#include <string>
#include <matheval.h>
#include <sstream>
#include "network_gel_relax_utils.hpp"
#include <assert.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string.h>
#include <regex>
#include <iomanip>

#define DEFAULT_MAX_STEPS 1000

using namespace std;

//Parse a job file and specifically extract a gel constitutive model
bool parse_input(string file_name, GelModel &gmod){

    ifstream input;
    string nextline;
    vector<string> tokens;
    double x, y, z, fine_mag;
    int n_coarse, n_fine, num_read;
    Vec3D coarse_disp, fine_disp;
    json parsed;

    input.open(file_name);
    if(! input.is_open()){
        cerr << "The job file could not be opened.\n";
        return false;
    }
    //Attempt to parse the specified file
    try{
        parsed = json::parse(input);
    }
    catch(exception &e){
        cerr << e.what() << endl;
        input.close();
        return false;
    }   

    input.close();

    //Constitutive model of the gel
    if(parsed.contains("gel")){
        if(! gmod.initialize(parsed["gel"])){
            cerr << "The gel model could not be parsed.\n";
            return false;
        }
    }

    return true;
}

//Given a gel constitutive model, use a Newton root finding method to identify
//the equilibrium swelling ratio.
double calc_equilibrium_swelling(GelModel gmod, int max_steps){

    int steps, count = gmod.param_values.size() + 1;
    double first_deriv, second_deriv, stretch, curr_val, prev_val, opt_stretch;
    void *f, *f_l, *f_ll;
    vector<char *> arg_names;
    vector<double> values;
    std::ostringstream oss1, oss2, oss3;
    std::regex i1_regex("I1");
    std::regex i2_regex("I2");
    std::regex j_regex("J");
    const char *var_name = "stretch";
    string original;

    //Initialize the guess for the search
    stretch = 2;

    //Replace the principal invariants with appropriate functions of the
    //common principal stretch for the case of isotropic swelling
    original = string(evaluator_get_string(gmod.energy_func));
    oss1 << std::regex_replace(original, i1_regex, "(3*stretch^2)");
    oss2 << std::regex_replace(oss1.str(), i2_regex, "(3*stretch^4)");
    oss3 << std::regex_replace(oss2.str(), j_regex, "stretch^3");
    f = evaluator_create((char *) oss3.str().c_str());
    assert(f);

    //Construct first partial derivatives
    f_l = evaluator_derivative(f, (char *) var_name);
    assert(f_l);
    f_ll = evaluator_derivative(f_l, (char *) var_name);
    assert(f_ll);

    arg_names.push_back((char *) var_name);
    for(int iter = 0; iter < count - 1; iter ++){
        arg_names.push_back(gmod.arg_names[iter+gmod.inv_funcs.size()]);
    }
    values.push_back(stretch);
    values.insert(values.end(), gmod.param_values.begin(), gmod.param_values.end());
    curr_val = evaluator_evaluate(f, count, &arg_names[0], &values[0]);

    //Perform Newton's method searches
    steps = 0;
    do{

        first_deriv = evaluator_evaluate(f_l, count, &arg_names[0], &values[0]);
        second_deriv = evaluator_evaluate(f_ll, count,&arg_names[0],&values[0]);
        steps ++;

        stretch = fmax(stretch - first_deriv / second_deriv, 1. + TOL);
        prev_val = curr_val;
        values[0] = stretch;
        curr_val = evaluator_evaluate(f, count, &arg_names[0], &values[0]);
        if(fabs(curr_val - prev_val) < TOL) break;
    } while(steps < max_steps && fabs(first_deriv) > TOL);

    evaluator_destroy(f);
    evaluator_destroy(f_l);
    evaluator_destroy(f_ll);

    return stretch;
}

//Given a job file, attempt to parse the gel constitutive model and identify
//the consequent equilibrium swelling value.
int main(int argc, char **argv){

    GelModel gmod;
    int num_read, max_steps;
    double pstretch;

    if(argc < 2){
        cerr << "Usage: job file, (optional) maximum steps\n"; 
    }

    if(argc >= 3){
        if( (num_read = sscanf(argv[2], "%d", &max_steps))){
            if(max_steps <= 0){
                cerr << "The second argument must be a positive integer.\n";
                cerr << "The default number of maximum steps will be used\n";
                max_steps = DEFAULT_MAX_STEPS;
            }
        }
        else {
            cerr << "The second argument should be a positive integer.\n";
            cerr << "The non-integer argument will be ignored.\n";
            max_steps = DEFAULT_MAX_STEPS;
        }
    }
    else max_steps = DEFAULT_MAX_STEPS;

    if(! parse_input(string(argv[1]), gmod)){
        cerr << "Parsing of the job file failed.\n";
    }

    pstretch = calc_equilibrium_swelling(gmod, max_steps);
    cout << setprecision(6) << scientific;
    cout << "Equilibrium swelling: " << pstretch*pstretch*pstretch << endl;
}
