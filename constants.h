#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include <stdio.h>
#include <chrono>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include <fstream>
#include<set>
#include <stdlib.h>
#include <time.h>
#include <random>

using namespace std;
using namespace std::chrono;
using namespace cub;
CachingDeviceAllocator g_allocator(true);
#define ull unsigned long long
const ull h_q = 65521;
const int h_p = 1024;
const ull h_n = 1 << 20;
const ull h_m = 13;
const ull h_mps[] = {7, 31, 127, 8191, 131071, 524287};
const int h_masksz = 6;
const ull h_shifts[] = {3, 5, 7, 13, 17, 19};
const ull h_cumShifts[] = {0, 3, 8, 15, 28, 45};
const ull h_ds[] = {2, 3, 5, 11, 13, 17};
const ull h_HTSZ = 1 << 18;
const int h_d = 2;
__constant__ const int d_d = 2;
__constant__ const ull d_q = 65521;
__constant__ const ull d_n = 1 << 20;
__constant__  const int d_p = 1024;
__constant__ const ull d_m = 13;
__constant__ const ull d_mps[] = {7, 31, 127, 8191, 131071, 524287};
__constant__ const int d_masksz = 6;
__constant__ const ull d_masks[] = {7, 248, 32512, 268402688, 35184103653376ull, 18446708889337462784ull};
__constant__ const ull d_nmasks[] = {18446744073709551608ull, 18446744073709551367ull, 18446744073709519103ull, 18446744073441148927ull, 18446708889605898239ull, 35184372088831ull};
__constant__ const ull d_shifts[] = {3, 5, 7, 13, 17, 19};
__constant__ const ull d_cumShifts[] = {0, 3, 8, 15, 28, 45};
__constant__ const ull d_ds[] = {2, 3, 5, 11, 13, 17};
__constant__ const ull d_HTMSK = h_HTSZ - 1;
char* g_h_data;
char* g_h_patterns;
char* g_d_data;
char* g_d_patterns;