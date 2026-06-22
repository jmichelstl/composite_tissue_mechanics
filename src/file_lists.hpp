#include <string>
#include <vector>

using namespace std;

vector<string> split(string input, char delim);

bool get_file_lists(string input, string tag, string extension, vector<string> &nmc_files, vector<string> &log_files, vector<string> &nmt_files, vector<string> &edat_files);
