
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
struct testcase {
private:
    string input_str;
    vector<pair<int, int>>expectedMatches;
    char* StringGeneration(int sz) {
        char* ret = new char[sz + 1];
        mt19937 generator{ random_device{}() };
        uniform_int_distribution<int> distribution{ 'a', 'a' + h_d - 1 };
        string rand_str(sz, '\0');
        for (auto& dis : rand_str)
            dis = distribution(generator);
        for (int i = 0; i < sz; i++) {
            ret[i] = rand_str[i];
        }
        ret[sz] = 0;
        return ret;
    }
    void WriteData(char* data) {
        ofstream myfile;
        myfile.open("outputfiles\\data.txt");
        myfile << string(data);
        myfile.close();
    }
    void WritePatterns(char * patterns) {
        set<string>st;
        ofstream myfile;
        myfile.open("outputfiles\\patterns.txt");
        for (int patternIndex = 0; patternIndex < h_p; patternIndex++) {
            string pattern = "";
            for (int i = patternIndex * h_m; i < patternIndex * h_m + h_m; i++)
                pattern += g_h_patterns[i];
            st.insert(pattern);
            myfile << patternIndex << ": " << pattern << endl;
        }
        if (st.size() != h_p) {
            cout << "Duplicate pattern occurred" << endl; 
        }
        myfile.close();
    }
    void WriteMatches(vector<pair<int, int>> matches, string fileName) {
        ofstream myfile;
        myfile.open(fileName.c_str());
        sort(matches.begin(), matches.end());
        myfile << "Row#\tPattern Index\tposition" << endl;
        for (int i = 0; i < matches.size(); i++) {
            myfile << i << ":\t"<<matches[i].first << "\t\t" << matches[i].second << endl;
        }
        myfile.close();
    }
    char * PatternsGeneration(){
        set<string>st;
        while(st.size() != h_p){
            char * ptrn = StringGeneration(h_m);
            st.insert(string(ptrn));
        }
        char * ret = new char[h_p * h_m + 1];
        ret[h_p * h_m] = 0;
        while(st.size()){
            for(int i = 0; i < h_m; i++){
                ret[h_m * (h_p - st.size()) + i] = (*st.begin())[i];
            }
            st.erase(st.begin());
        }
        return ret;
    }
    void GenerateInputData() {
        g_h_data = StringGeneration(h_n);
        g_d_data = NULL;
        g_h_patterns = PatternsGeneration();
        g_d_patterns = NULL;
    }
    void FindPattern(int patternIndex) {
        string pattern = "";
        for (int i = patternIndex * h_m; i < patternIndex * h_m + h_m; i++)
            pattern += g_h_patterns[i];

        size_t pos = input_str.find(pattern);
        while (pos != string::npos) {
            expectedMatches.push_back({ patternIndex, pos });
            pos = input_str.find(pattern, pos + 1);
        }

    }
    void SolveOnCPU() {
        input_str = string(g_h_data);
        for (int i = 0; i < h_p; i++) 
            FindPattern(i);
    }
public:
    testcase() {
        GenerateInputData();
        WriteData(g_h_data);
        WritePatterns(g_h_patterns);
        SolveOnCPU();
        WriteMatches(expectedMatches, "outputfiles\\Expected.txt");
    }
    static void CopyDataToDevice(){
        CubDebugExit(g_allocator.DeviceAllocate((void**)&g_d_data, sizeof(char) * h_n));
        CubDebugExit(cudaMemcpy(g_d_data, g_h_data, sizeof(char) * h_n, cudaMemcpyHostToDevice));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&g_d_patterns, sizeof(char) * h_p * h_m));
        CubDebugExit(cudaMemcpy(g_d_patterns, g_h_patterns, sizeof(char) * h_p * h_m, cudaMemcpyHostToDevice));
        // debug
        //delete[] g_h_patterns;
        //delete[] g_h_data;
    }
    void Validate(int* h_output) {
        vector<pair<int, int>>gpuMatches;
        for (int i = 0; i < h_n; i++) {
            if (h_output[i] != -1) {
                gpuMatches.push_back({h_output[i], i});
            }
        }
        sort(gpuMatches.begin(), gpuMatches.end());
        WriteMatches(gpuMatches, "outputfiles\\Actual.txt");
        bool same = true;
        if (gpuMatches.size() != expectedMatches.size()) {
            cout << "sizes are not equal expected size is " << expectedMatches.size() << " and actual size is " << gpuMatches.size() << endl;
            same = false;
        }
        for (int i = 0, limit = 0;i < min(gpuMatches.size(), expectedMatches.size()); i++) {
            if (gpuMatches[i].first != expectedMatches[i].first || gpuMatches[i].second != expectedMatches[i].second) {
                cout << "Mismatch at position: " << i << endl;
                limit++;
                same = false;
                if(limit >= 100){
                    break;
                }
            }
        }
        if (same) {
            cout << "Code works fine" << endl;
        }
        else {
            cout << "Results doesn't match, debug your code" << endl;
        }
    }

}test;
struct CustomSum
{
    CUB_RUNTIME_FUNCTION __host__ __device__ __forceinline__
        int operator()(const int& a, const int& b) const {
        return (a + b) % d_q;
    }
}sumMod;
struct CustomSumNew
{
    CUB_RUNTIME_FUNCTION __device__ __forceinline__
        ull operator()(const ull& a, const ull& b) const {
            ull ans = 0;
            for(int j = 0; j < d_masksz; j++){
                ull sum = ((a & d_masks[j]) >> d_cumShifts[j]) + ((b & d_masks[j]) >> d_cumShifts[j]);
                sum = (sum & d_mps[j]) + (sum >> d_shifts[j]);
                sum = sum >= d_mps[j] ? sum - d_mps[j] : sum;
                ans |= sum << d_cumShifts[j];
            }
        return ans;
    }
}sumModMersennePrime;
__global__ void CalculateHashPattern(char* d_patterns, int* d_controlArray, int* d_hashTable) {
    int patternIndex = threadIdx.x, patternHash = 0;

    for (int i = patternIndex * d_m; i < patternIndex * d_m + d_m; i++)
        patternHash = (patternHash * d_d + (d_patterns[i] - 'a' + 1)) % d_q;

    while (atomicAdd(&d_controlArray[patternHash], 1) != 0)
        patternHash = (patternHash + 1) % d_q;

    d_hashTable[patternHash] = patternIndex;
}
__global__ void CalculateHashPatternNew(char* d_patterns, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
    int patternIndex = threadIdx.x;
    ull patternHash = 0;

    for (int i = patternIndex * d_m; i < patternIndex * d_m + d_m; i++){
        for(int j = 0; j < d_masksz; j++){
            ull hash = (patternHash & d_masks[j]) >> d_cumShifts[j];
            hash = hash * d_ds[j] + (ull)(d_patterns[i] - 'a' + 1);
            hash = (hash & d_mps[j]) + (hash >> d_shifts[j]);
            hash = hash >= d_mps[j] ? hash - d_mps[j] : hash;
            patternHash &= d_nmasks[j];
            patternHash |= hash << d_cumShifts[j];
        }
    }

    d_patternHashes[patternIndex] = patternHash;
    while (atomicAdd(&d_controlArray[patternHash & d_HTMSK], 1) != 0)
        patternHash = (patternHash == d_HTMSK) ? 0: patternHash + 1;

    d_hashTable[patternHash & d_HTMSK] = patternIndex;
}
__global__ void CalculateHashes(int* d_a, char* d_data, int* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = (d_lookupTable[(d_n - i - 1) % (d_q - 1)] * (d_data[i] - 'a' + 1)) % d_q;
}
__global__ void CalculateHashesNew(ull* d_a, char* d_data, ull* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = 0;
    for(int j = 0; j < d_masksz; j++){
        ull hash = (((d_lookupTable[(d_n - i - 1) % (d_mps[j] - 1)] & d_masks[j]) >> d_cumShifts[j]) * (ull)(d_data[i] - 'a' + 1)); //hash =  (ds[j]^(i mod (mps[j] - 1)) % mps[j]) * data[i]
        hash = (hash & d_mps[j]) + (hash >> d_shifts[j]);
        hash = hash >= d_mps[j] ? hash - d_mps[j] : hash;
        d_a[i] |= hash << d_cumShifts[j];
    }
    
}
__global__ void FindMatches(int* d_prefixSum, char* d_data, char* d_patterns, int* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output) {
    ull j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + d_m - 1 <= d_n) {
        int hash = ((((ull)(d_prefixSum[j + d_m - 1] - (j ? d_prefixSum[j - 1] : 0)) + d_q) % d_q) * (ull)d_lookupTable[(d_m + ((d_n - j + d_q - 2ll) / (d_q - 1ll)) * (d_q - 1ll) - d_n + j) % (d_q - 1ll)]) % d_q;
        while (d_controlArray[hash]) {
            int patternIndex = d_hashTable[hash];
            bool match = true;
            for (int i = patternIndex * d_m, offset = 0; i < patternIndex * d_m + d_m; i++, offset++) {
                if (d_patterns[i] != d_data[j + offset]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                d_output[j] = patternIndex;
                return; 
            }
            hash = (hash + 1) % d_q;
        }
    }
}
__global__ void FindMatchesNew(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output, ull* d_patternHashes) {
    ull j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + d_m - 1 <= d_n) {
        ull hash = 0;
        for(int k = 0; k < d_masksz; k++){
            ull tmp = (d_prefixSum[j + d_m - 1] & d_masks[k]) >> d_cumShifts[k];
            if(j){
                tmp = (tmp + d_mps[k]) - ((d_prefixSum[j - 1] & d_masks[k]) >> d_cumShifts[k]);
                tmp = (tmp >= d_mps[k] ? tmp - d_mps[k] : tmp);
            }
            tmp = tmp * ((d_lookupTable[(d_m + ((d_n - j + d_mps[k] - 2ll) / (d_mps[k] - 1ll)) * (d_mps[k] - 1ll) - d_n + j) % (d_mps[k] - 1ll)] & d_masks[k]) >> d_cumShifts[k]) ;
            tmp = (tmp & d_mps[k]) + (tmp >> d_shifts[k]);
            hash |= (tmp >= d_mps[k] ? tmp - d_mps[k] : tmp) << d_cumShifts[k]; 
        }
        for(int i = 0; i < d_p; i++){
            if (hash == d_patternHashes[i]) {
                d_output[j] = i;
                return;
            }
        }
        /*
        while (d_controlArray[hash & d_HTMSK]) {
            if (hash == d_patternHashes[d_hashTable[hash & d_HTMSK]]) {
                d_output[j] = d_hashTable[hash & d_HTMSK];
                return;
            }
            hash = (hash == d_HTMSK) ? 0: hash + 1;
        }
        */
    }
}
class PaperImplementation{
private:
    static int* Step1() {
        int* ret = NULL;
        int *h_lookupTabe = new int [h_q];
        for (int i = 0, current = 1; i < h_q; i++, current = (current * h_d) % h_q ) 
            h_lookupTabe[i] = current;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&ret, sizeof(int) * h_q));
        CubDebugExit(cudaMemcpy(ret, h_lookupTabe, sizeof(int) * h_q, cudaMemcpyHostToDevice));
        delete[] h_lookupTabe;
        cudaDeviceSynchronize();
        return ret;
    }
    static pair<int*, int*> Step2(char * d_patterns) {
        int* d_controlArray = NULL, * d_hashTable = NULL, * h_controlArray = new int[h_q], * h_hashTable = new int[h_q];
        for (int i = 0; i < h_q; i++) {
            h_controlArray[i] = 0;
            h_hashTable[i] = -1;
        }
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_controlArray, sizeof(int) * h_q));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hashTable, sizeof(int) * h_q));
        CubDebugExit(cudaMemcpy(d_controlArray, h_controlArray, sizeof(int) * h_q, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_hashTable, h_hashTable, sizeof(int) * h_q, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        CalculateHashPattern <<< 1, h_p >>> (d_patterns, d_controlArray, d_hashTable);
        cudaDeviceSynchronize();
        delete[] h_hashTable;
        delete[] h_controlArray;
        return { d_controlArray, d_hashTable };
    }
    static int* Step3(char * d_data, int* d_lookupTable) {
        int* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(int) * h_n));
        CalculateHashes << <h_n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaDeviceSynchronize();
        return d_a;
    }
    static int* Step4(int* d_a) {
        int* d_prefixSum = NULL;
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_prefixSum, sizeof(int) * h_n));
        cudaDeviceSynchronize();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumMod, h_n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumMod, h_n));
        cudaFree(d_a);
        cudaFree(d_temp_storage);
        return d_prefixSum;
    }
    static int* Step5(int* d_prefixSum, char* d_data, char* d_patterns, int* d_lookupTable, int* d_controlArray, int* d_hashTable) {
        int* h_output = new int [h_n];
        for (int i = 0; i < h_n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * h_n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * h_n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        FindMatches << <h_n / 256, 256 >> > (d_prefixSum, d_data, d_patterns, d_lookupTable, d_controlArray, d_hashTable, d_output);
        CubDebugExit(cudaMemcpy(h_output, d_output, sizeof(int) * h_n, cudaMemcpyDeviceToHost));
        cudaFree(d_output);
        cudaFree(d_prefixSum);
        cudaFree(d_data);
        cudaFree(d_patterns);
        cudaFree(d_lookupTable);
        cudaFree(d_controlArray);
        cudaFree(d_hashTable);
        return h_output;
    }
public:
    static int* Execute() {
        //CopyDataToDevice();
        //1.Load a preprocessed lookup table for di mod q (0 ≤ i ≤ q − 1)
        int* d_lookupTable = Step1();
        //2. Compute the values of h(Pk) for all k (0 ≤ k ≤ p − 1) in parallel and create the hash table HT using the calculated values.
        pair<int*, int*> p = Step2(g_d_patterns);
        //3.Compute the a0, a1,..., an−1 in parallel.
        int* d_a = Step3(g_d_data, d_lookupTable);
        //4.Compute the prefix-sums ˆa0, aˆ1,..., aˆn−1.
        int* d_prefixSum = Step4(d_a);
        //5.  For all j (0 ≤ j ≤ n − m), compute ( ˆaj+m−1 − aˆ j−1) · dm−n−j, which is equal to h(tjtj + 1 ... tj + m−1).If array control[h(tjtj + 1 ... tj + m−1)]  0 then compare the characters of text and pattern with Match(i, j).
        int* h_output = Step5(d_prefixSum, g_d_data, g_d_patterns, d_lookupTable, p.first, p.second);
    
        return h_output;
    }
};
class ProposedImplementation{
private:
    static ull* Step1(){
        ull* ret = NULL;
        ull* h_lookupTabe = new ull [h_mps[5]];
        ull currents[] = {1, 1, 1, 1, 1, 1};
        for(int i = 0; i < h_mps[5]; i++){
            h_lookupTabe[i] = 0;
            for(int j = 0; j <h_masksz; j++){
                h_lookupTabe[i] |= currents[j] << h_cumShifts[j];
                currents[j] = currents[j] * h_ds[j];
                currents[j] = (currents[j] & h_mps[j]) + (currents[j] >> h_shifts[j]);
                currents[j] = currents[j] >= h_mps[j] ? currents[j] - h_mps[j] : currents[j];
            }
            
        }    
        CubDebugExit(g_allocator.DeviceAllocate((void**)&ret, sizeof(ull) * h_mps[5]));
        CubDebugExit(cudaMemcpy(ret, h_lookupTabe, sizeof(ull) * h_mps[5], cudaMemcpyHostToDevice));
        delete[] h_lookupTabe;
        cudaDeviceSynchronize();
        return ret;
    }
    static pair<pair<int*, int*>, ull*> Step2(char * d_patterns) {
        int* d_controlArray = NULL, * d_hashTable = NULL,* h_controlArray = new int[h_HTSZ],* h_hashTable = new int [h_HTSZ];
        ull * d_patternHashes = NULL;
        for (int i = 0; i < h_HTSZ; i++) {
            h_controlArray[i] = 0;
            h_hashTable[i] = -1;
        }
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_patternHashes, sizeof(ull) * h_p));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_controlArray, sizeof(int) * h_HTSZ));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hashTable, sizeof(int) * h_HTSZ));
        CubDebugExit(cudaMemcpy(d_controlArray, h_controlArray, sizeof(int) * h_HTSZ, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_hashTable, h_hashTable, sizeof(int) * h_HTSZ, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        CalculateHashPatternNew <<< 1, h_p >>> (d_patterns, d_controlArray, d_hashTable, d_patternHashes);
        cudaFree(d_patterns);
        cudaDeviceSynchronize();
        delete[] h_hashTable;
        delete[] h_controlArray;
        return { {d_controlArray, d_hashTable}, d_patternHashes};
    }
    static ull* Step3(char * d_data, ull* d_lookupTable) {
        ull* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(ull) * h_n));
        CalculateHashesNew << <h_n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaFree(d_data);
        cudaDeviceSynchronize();
        return d_a;
    }
    static ull* Step4(ull* d_a) {
        ull* d_prefixSum = NULL;
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_prefixSum, sizeof(ull) * h_n));
        cudaDeviceSynchronize();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumModMersennePrime, h_n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumModMersennePrime,h_n));
        cudaFree(d_a);
        cudaFree(d_temp_storage);
        return d_prefixSum;
    }
    static int* Step5(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
        int* h_output = new int [h_n];
        for (int i = 0; i < h_n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * h_n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * h_n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        FindMatchesNew << <h_n / 256, 256 >> > (d_prefixSum, d_lookupTable, d_controlArray, d_hashTable, d_output, d_patternHashes);
        CubDebugExit(cudaMemcpy(h_output, d_output, sizeof(int) * h_n, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        cudaFree(d_output);
        cudaFree(d_prefixSum);
        cudaFree(d_lookupTable);
        cudaFree(d_controlArray);
        cudaFree(d_hashTable);
        cudaFree(d_patternHashes);
        return h_output;
    }
public:
    static int* Execute() {
        ull* d_lookupTable = Step1();
        cout << "step 1 done" << endl;
        static pair<pair<int*, int*>, ull*> p = Step2(g_d_patterns);
        cout << "step 2 done" << endl;
        ull* d_a = Step3(g_d_data, d_lookupTable);
        cout << "step 3 done" << endl;
        ull* d_prefixSum = Step4(d_a);
        cout << "step 4 done" << endl;
        int* h_output = Step5(d_prefixSum, d_lookupTable, p.first.first, p.first.second, p.second);
        cout << "step 5 done" << endl;
        return h_output;
    }
};
int main(){
    testcase::CopyDataToDevice();
    test.Validate(ProposedImplementation::Execute());
    return 0;
}
