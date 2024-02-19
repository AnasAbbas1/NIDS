#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <set>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <ctime>
#include <unordered_map>
#include <cassert>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>

using namespace std;
using namespace std::chrono;
using namespace cub;
CachingDeviceAllocator g_allocator(true);
mt19937 generator{ random_device{}() };
#define ull unsigned long long
#define usi unsigned short int
#define paper_prime 65521
#define paper_prime_1 65520
#define ascii_code_1 96
#define four_primes_1 8190
const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
ull h_n = 1 << 26;
int h_p = 1 << 10;
ull h_m = 15;
const unsigned int h_mps[] = {7, 31, 127, 8191, 131071, 524287};
const unsigned int h_mps_1[] = {6, 30, 126, 8190, 131070, 524286};
const usi h_masksz = 6;
const usi h_shifts[] = {3, 5, 7, 13, 17, 19};
const usi h_cumShifts[] = {0, 3, 8, 15, 28, 45};
const usi h_oppShifts[] = {0, 3, 5, 7, 13, 17};
ull h_ds[] = {2, 4, 6, 8, 10, 12};
const ull h_HTSZ = 1 << 18;
int h_d = 2;
const ull h_prime = *lower_bound(primes, primes + sizeof(primes) / sizeof(int) , h_d);
__constant__ const ull d_prime = 2;
__constant__ ull d_n;
__constant__ ull d_m;
__constant__ unsigned int d_mps[h_masksz];
__constant__ unsigned int d_mps_1[h_masksz];
__constant__ const usi d_masksz = h_masksz;
__constant__ const ull d_masks[] = {7, 248, 32512, 268402688, 35184103653376ull, 18446708889337462784ull};
__constant__ const ull d_nmasks[] = {18446744073709551608ull, 18446744073709551367ull, 18446744073709519103ull, 18446744073441148927ull, 18446708889605898239ull, 35184372088831ull};
__constant__ usi d_shifts[h_masksz];
__constant__ usi d_cumShifts[h_masksz];
__constant__ usi d_oppShifts[h_masksz];
__constant__ ull d_ds[h_masksz];
__constant__ const ull d_HTMSK = h_HTSZ - 1;
__constant__ const int d_paper_c = 1074528000;
__constant__ const ull d_proposed_c = 562801779703800ull;
char* g_h_data;
char* g_h_patterns;
char* g_d_data;
char* g_d_patterns;
clock_t timer[2][5];
clock_t cpu;
string output_path;
bool write_data;
bool correctResult;
ofstream outputFile;
string curCase;
void SetConstants(int argc, char *argv[]){
    if (argc != 1){
        assert(argc >= 5);
        h_n = 1 << stoi(string(argv[1]));
        h_p = 1 << stoi(string(argv[2]));
        h_m = stoi(string(argv[3]));
        h_d = stoi(string(argv[4]));
        for (int i = 0; i < h_masksz; i++){
            uniform_int_distribution<int> distribution{ h_d, (int) h_mps[i] - 1};
            h_ds[i] = distribution(generator);
        }
        if (argc == 6){
            write_data = (argv[4][0] == 'w') || (argv[4][0] == 'W');
        }
    }
    char buffer[1 << 10];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        perror("getcwd() error");
        return;
    }
    string cwd = string(argv[0]);
    for(int i = cwd.size() - 2; i >= 0; i--){
        if(cwd[i] == '/'){
            cwd = cwd.substr(0, i);
            break;
        }
    }
    output_path = string(buffer) + "/" + cwd + "/outputfiles/";
    mkdir(output_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    curCase = "n=" + string(argv[1]) + "_k=" + string(argv[2]) + "_m=" + string(argv[3]);
    output_path += curCase + "/";
    assert (mkdir(output_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);
    outputFile = ofstream(string(output_path + "output.txt").c_str());
}
void CopyConstants(){
    CubDebugExit(cudaMemcpyToSymbol(d_n, (void *)&h_n, sizeof(ull)));
    CubDebugExit(cudaMemcpyToSymbol(d_m, (void *)&h_m, sizeof(ull)));
    CubDebugExit(cudaMemcpyToSymbol(d_mps, h_mps, sizeof(h_mps)));
    CubDebugExit(cudaMemcpyToSymbol(d_mps_1, h_mps_1, sizeof(h_mps)));
    CubDebugExit(cudaMemcpyToSymbol(d_shifts, h_shifts, sizeof(h_shifts)));
    CubDebugExit(cudaMemcpyToSymbol(d_cumShifts, h_cumShifts, sizeof(h_cumShifts)));
    CubDebugExit(cudaMemcpyToSymbol(d_oppShifts, h_oppShifts, sizeof(h_oppShifts)));
    CubDebugExit(cudaMemcpyToSymbol(d_ds, h_ds, sizeof(h_ds)));
}

void FinalReport(){
    outputFile << "proposed implementation is  " << timer[0][4] / (double)timer[1][4] << " times faster than paper implementation" << endl;
    outputFile << "proposed implementation is  " << cpu / (double)timer[1][4] << " times faster than cpu" << endl;
    outputFile << "paper implementation is " <<  cpu / (double)timer[0][4] << " times faster than cpu" << endl;
    outputFile.close();
    cout << "output of " << curCase << " were " << (correctResult ? "correct" : "incorrect") << endl;
}
