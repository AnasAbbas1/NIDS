#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include <stdio.h>
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

using namespace std;
using namespace std::chrono;
using namespace cub;
CachingDeviceAllocator g_allocator(true);
#define ull unsigned long long
const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
const ull h_q = 65521;
const int h_p = 1;
const ull h_n = 1 << 27;
const ull h_m = 1000;
const ull h_mps[] = {8191, 131071, 524287};
const int h_masksz = 3;
const ull h_shifts[] = {13, 17, 19};
const ull h_cumShifts[] = {0, 13, 30};
const ull h_ds[] = {2, 3, 5};
const ull h_HTSZ = 1 << 18;
const int h_d = 1;
const ull h_prime = *lower_bound(primes, primes + sizeof(primes) / sizeof(int) , h_d);
__constant__ const ull d_prime = 2;
__constant__ const ull d_q = h_q;
__constant__ const ull d_n = h_n;
__constant__  const int d_p = h_p;
__constant__ const ull d_m = h_m;
__constant__ const ull d_mps[] = {8191, 131071, 524287};
__constant__ const int d_masksz = h_masksz;
__constant__ const ull d_masks[] = {8191, 1073733632ull, 562948879679488ull};
__constant__ const ull d_nmasks[] = {18446744073709543424ull, 18446744072635817983ull, 18446181124829872127ull};
__constant__ const ull d_shifts[] = {13, 17, 19};
__constant__ const ull d_cumShifts[] = {0, 13, 30};
__constant__ const ull d_ds[] = {2, 3, 5};
__constant__ const ull d_HTMSK = h_HTSZ - 1;
char* g_h_data;
char* g_h_patterns;
char* g_d_data;
char* g_d_patterns;
clock_t timer[2][5];
const string output_path = "/content/NIDS-stress_test-fast/outputfiles/";
