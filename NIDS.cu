
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
__constant__ const int d_d = 2;
__constant__ const ull d_q = 65521;
__constant__ const int d_p = 1024;
__constant__ const ull d_n = 1 << 18;
__constant__ const ull d_m = 13;
__constant__ const ull d_mps[] = {7, 31, 127, 8191, 131071, 524287};
__constant__ const int d_masksz = 6;
__constant__ const ull d_masks[] = {7, 248, 32512, 268402688, 35184103653376ull, 18446708889337462784ull};
__constant__ const ull d_nmasks[] = {18446744073709551608ull, 18446744073709551367ull, 18446744073709519103ull, 18446744073441148927ull, 18446708889605898239ull, 35184372088831ull};
__constant__ const ull d_shifts[] = {3, 5, 7, 13, 17, 19};
__constant__ const ull d_cumShifts[] = {0, 3, 8, 15, 28, 45};
__constant__ const ull d_ds[] = {2, 3, 5, 11, 13, 17};
__constant__ const ull d_HTSZ = 1 << 18;
__constant__ const ull d_HTMSK = HTSZ - 1;
const int h_d = 2;
const ull h_q = 65521;
const int h_p = 1024;
const ull h_n = 1 << 18;
const ull h_m = 13;
const ull h_mps[] = {7, 31, 127, 8191, 131071, 524287};
const int h_masksz = 6;
const ull h_masks[] = {7, 248, 32512, 268402688, 35184103653376ull, 18446708889337462784ull};
const ull h_nmasks[] = {18446744073709551608ull, 18446744073709551367ull, 18446744073709519103ull, 18446744073441148927ull, 18446708889605898239ull, 35184372088831ull};
const ull h_shifts[] = {3, 5, 7, 13, 17, 19};
const ull h_cumShifts[] = {0, 3, 8, 15, 28, 45};
const ull h_ds[] = {2, 3, 5, 11, 13, 17};
const ull h_HTSZ = 1 << 18;
const ull h_HTMSK = HTSZ - 1;
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
        for (int patternIndex = 0; patternIndex < p; patternIndex++) {
            string pattern = "";
            for (int i = patternIndex * m; i < patternIndex * m + m; i++)
                pattern += h_patterns[i];
            st.insert(pattern);
            myfile << patternIndex << ": " << pattern << endl;
        }
        if (st.size() != p) {
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
        while(st.size() != p){
            char * ptrn = StringGeneration(m);
            st.insert(string(ptrn));
        }
        char * ret = new char[p * m + 1];
        ret[p * m] = 0;
        while(st.size()){
            for(int i = 0; i < m; i++){
                ret[m * (p - st.size()) + i] = (*st.begin())[i];
            }
            st.erase(st.begin());
        }
        return ret;
    }
    void GenerateInputData() {
        h_data = StringGeneration(n);
        d_data = NULL;
        h_patterns = PatternsGeneration();
        d_patterns = NULL;
    }
    void FindPattern(int patternIndex) {
        string pattern = "";
        for (int i = patternIndex * m; i < patternIndex * m + m; i++)
            pattern += h_patterns[i];

        size_t pos = input_str.find(pattern);
        while (pos != string::npos) {
            expectedMatches.push_back({ patternIndex, pos });
            pos = input_str.find(pattern, pos + 1);
        }

    }
    void SolveOnCPU() {
        input_str = string(h_data);
        for (int i = 0; i < p; i++) 
            FindPattern(i);
    }
    void CopyDataToDevice(){
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(char) * n));
        CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(char) * n, cudaMemcpyHostToDevice));
        CubDebugExit(g_allocator.DeviceAllocate((void**)d_patterns, sizeof(char) * p * m));
        CubDebugExit(cudaMemcpy(d_patterns, h_patterns, sizeof(char) * p * m, cudaMemcpyHostToDevice));
        delete[] h_patterns;
        delete[] h_data;
    }
public:
    char* h_data;
    char* h_patterns;
    char* d_data;
    char* d_patterns;
    testcase() {
        GenerateInputData();
        WriteData(h_data);
        WritePatterns(h_patterns);
        SolveOnCPU();
        WriteMatches(expectedMatches, "outputfiles\\Expected.txt");
        CopyDataToDevice();
    }

    void Validate(int* h_output) {
        vector<pair<int, int>>gpuMatches;
        for (int i = 0; i < n; i++) {
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
        for (int i = 0;i < min(gpuMatches.size(), expectedMatches.size()); i++) {
            if (gpuMatches[i].first != expectedMatches[i].first || gpuMatches[i].second != expectedMatches[i].second) {
                cout << "Mismatch at position: " << i << endl;
                same = false;
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
        return (a + b) % q;
    }
}sumMod;
struct CustomSumNew
{
    CUB_RUNTIME_FUNCTION __host__ __device__ __forceinline__
        ull operator()(const ull& a, const ull& b) const {
            ull ans = 0;
            for(int j = 0; j < masksz; j++){
                ull sum = ((a & masks[j]) >> cumShifts[j]) + ((b & masks[j]) >> cumShifts[j]);
                sum = (sum & mps[j]) + (sum >> shifts[j]);
                sum = sum >= mps[j] ? sum - mps[j] : sum;
                ans |= sum << cumShifts[j];
            }
        return ans;
    }
}sumModMersennePrime;
__global__ void CalculateHashPattern(char* d_patterns, int* d_controlArray, int* d_hashTable) {
    int patternIndex = threadIdx.x, patternHash = 0;

    for (int i = patternIndex * m; i < patternIndex * m + m; i++)
        patternHash = (patternHash * d + (d_patterns[i] - 'a' + 1)) % q;

    while (atomicAdd(&d_controlArray[patternHash], 1) != 0)
        patternHash = (patternHash + 1) % q;

    d_hashTable[patternHash] = patternIndex;
}
__global__ void CalculateHashPatternNew(char* d_patterns, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
    int patternIndex = threadIdx.x;
    ull patternHash = 0;

    for (int i = patternIndex * m; i < patternIndex * m + m; i++){
        for(int j = 0; j < masksz; j++){
            ull hash = (patternHash & masks[j]) >> cumShifts[j];
            hash = hash * ds[j] + (ull)(d_patterns[i] - 'a' + 1);
            hash = (hash & mps[j]) + (hash >> shifts[j]);
            hash = hash >= mps[j] ? hash - mps[j] : hash;
            patternHash &= nmasks[j];
            patternHash |= hash << cumShifts[j];
        }
    }

    d_patternHashes[patternIndex] = patternHash;
    while (atomicAdd(&d_controlArray[patternHash & HTMSK], 1) != 0)
        patternHash = (patternHash == HTMSK) ? 0: patternHash + 1;

    d_hashTable[patternHash & HTMSK] = patternIndex;
}
__global__ void CalculateHashes(int* d_a, char* d_data, int* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = (d_lookupTable[(n - i - 1) % (q - 1)] * (d_data[i] - 'a' + 1)) % q;
}
__global__ void CalculateHashesNew(ull* d_a, char* d_data, ull* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = 0;
    for(int j = 0; j < masksz; j++){
        ull hash = (((d_lookupTable[(n - i - 1) % (mps[j] - 1)] & masks[j]) >> cumShifts[j]) * (ull)(d_data[i] - 'a' + 1)); //hash =  (ds[j]^(i mod (mps[j] - 1)) % mps[j]) * data[i]
        hash = (hash & mps[j]) + (hash >> shifts[j]);
        hash = hash >= mps[j] ? hash - mps[j] : hash;
        d_a[i] |= hash << cumShifts[j];
    }
    
}
__global__ void FindMatches(int* d_prefixSum, char* d_data, char* d_patterns, int* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output) {
    ull j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + m - 1 <= n) {
        int hash = ((((ull)(d_prefixSum[j + m - 1] - (j ? d_prefixSum[j - 1] : 0)) + q) % q) * (ull)d_lookupTable[(m + ((n - j + q - 2ll) / (q - 1ll)) * (q - 1ll) - n + j) % (q - 1ll)]) % q;
        while (d_controlArray[hash]) {
            int patternIndex = d_hashTable[hash];
            bool match = true;
            for (int i = patternIndex * m, offset = 0; i < patternIndex * m + m; i++, offset++) {
                if (d_patterns[i] != d_data[j + offset]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                d_output[j] = patternIndex;
                return; 
            }
            hash = (hash + 1) % q;
        }
    }
}
__global__ void FindMatchesNew(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output, ull* d_patternHashes) {
    ull j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + m - 1 <= n) {
        ull hash = 0;
        for(int k = 0; k < masksz; k++){
            ull tmp = (d_prefixSum[j + m - 1] & masks[k]) >> cumShifts[k];
            if(j){
                tmp = tmp + mps[k] - ((d_prefixSum[j - 1] & masks[k]) >> cumShifts[k]);
                tmp = (tmp >= mps[k] ? tmp - mps[k] : tmp);
            }
            tmp = tmp * (d_lookupTable[(m + ((n - j + mps[k] - 2ll) / (mps[k] - 1ll)) * (mps[k] - 1ll) - n + j) % (mps[k] - 1ll)] >> cumShifts[k]) ;
            tmp = (tmp & mps[k]) + (tmp >> shifts[k]);
            hash |= (tmp >= mps[k] ? tmp - mps[k] : tmp) << cumShifts[k]; 
        }
        
        while (d_controlArray[hash & HTMSK]) {
            if (hash == d_patternHashes[d_hashTable[hash & HTMSK]]) {
                d_output[j] = d_hashTable[hash & HTMSK];
                return;
            }
            hash = (hash == HTMSK) ? 0: hash + 1;
        }
    }
}
class PaperImplementation{
private:
    static int* Step1() {
        int* ret = NULL;
        int *h_lookupTabe = new int [q];
        for (int i = 0, current = 1; i < q; i++, current = (current * d) % q ) 
            h_lookupTabe[i] = current;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&ret, sizeof(int) * q));
        CubDebugExit(cudaMemcpy(ret, h_lookupTabe, sizeof(int) * q, cudaMemcpyHostToDevice));
        delete[] h_lookupTabe;
        cudaDeviceSynchronize();
        return ret;
    }
    static pair<int*, int*> Step2(char * d_patterns) {
        int* d_controlArray = NULL, * d_hashTable = NULL,* h_controlArray = new int[q],* h_hashTable = new int [q];
        for (int i = 0; i < q; i++) {
            h_controlArray[i] = 0;
            h_hashTable[i] = -1;
        }
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_controlArray, sizeof(int) * q));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hashTable, sizeof(int) * q));
        CubDebugExit(cudaMemcpy(d_controlArray, h_controlArray, sizeof(int) * q, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_hashTable, h_hashTable, sizeof(int) * q, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        CalculateHashPattern <<< 1, p >>> (d_patterns, d_controlArray, d_hashTable);
        cudaDeviceSynchronize();
        delete[] h_hashTable;
        delete[] h_controlArray;
        return { d_controlArray, d_hashTable };
    }
    static int* Step3(char * d_data, int* d_lookupTable) {
        int* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(int) * n));
        CalculateHashes << <n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaDeviceSynchronize();
        return d_a;
    }
    static int* Step4(int* d_a) {
        int* d_prefixSum = NULL;
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_prefixSum, sizeof(int) * n));
        cudaDeviceSynchronize();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumMod, n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumMod, n));
        cudaFree(d_a);
        cudaFree(d_temp_storage);
        return d_prefixSum;
    }
    static int* Step5(int* d_prefixSum, char* d_data, char* d_patterns, int* d_lookupTable, int* d_controlArray, int* d_hashTable) {
        int* h_output = new int [n];
        for (int i = 0; i < n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        FindMatches << <n / 256, 256 >> > (d_prefixSum, d_data, d_patterns, d_lookupTable, d_controlArray, d_hashTable, d_output);
        CubDebugExit(cudaMemcpy(h_output, d_output, sizeof(int) * n, cudaMemcpyDeviceToHost));
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
        //1.Load a preprocessed lookup table for di mod q (0 ≤ i ≤ q − 1)
        int* d_lookupTable = Step1();
        //2. Compute the values of h(Pk) for all k (0 ≤ k ≤ p − 1) in parallel and create the hash table HT using the calculated values.
        pair<int*, int*> p = Step2(test.d_patterns);
        //3.Compute the a0, a1,..., an−1 in parallel.
        int* d_a = Step3(test.d_data, d_lookupTable);
        //4.Compute the prefix-sums ˆa0, aˆ1,..., aˆn−1.
        int* d_prefixSum = Step4(d_a);
        //5.  For all j (0 ≤ j ≤ n − m), compute ( ˆaj+m−1 − aˆ j−1) · dm−n−j, which is equal to h(tjtj + 1 ... tj + m−1).If array control[h(tjtj + 1 ... tj + m−1)]  0 then compare the characters of text and pattern with Match(i, j).
        int* h_output = Step5(d_prefixSum, test.d_data, test.d_patterns, d_lookupTable, p.first, p.second);
    
        return h_output;
    }
};
class ProposedImplementation{
private:
    static ull* Step1(){
        ull* ret = NULL;
        ull* h_lookupTabe = new ull [mps[5]];
        ull currents[] = {1, 1, 1, 1, 1, 1};
        for(int i = 0; i < mps[5]; i++){
            h_lookupTabe[i] = 0;
            for(int j = 0; j < masksz; j++){
                h_lookupTabe[i] |= currents[j] << cumShifts[j];
                currents[j] = currents[j] * ds[j];
                currents[j] = (currents[j] & mps[j]) + (currents[j] >> shifts[j]);
                currents[j] = currents[j] >= mps[j] ? currents[j] - mps[j] : currents[j];
            }
            
        }    
        CubDebugExit(g_allocator.DeviceAllocate((void**)&ret, sizeof(ull) * mps[5]));
        CubDebugExit(cudaMemcpy(ret, h_lookupTabe, sizeof(ull) * mps[5], cudaMemcpyHostToDevice));
        delete[] h_lookupTabe;
        cudaDeviceSynchronize();
        return ret;
    }
    static pair<pair<int*, int*>, ull*> Step2(char * d_patterns) {
        int* d_controlArray = NULL, * d_hashTable = NULL,* h_controlArray = new int[HTSZ],* h_hashTable = new int [HTSZ];
        ull * d_patternHashes = NULL;
        for (int i = 0; i < HTSZ; i++) {
            h_controlArray[i] = 0;
            h_hashTable[i] = -1;
        }
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_patternHashes, sizeof(ull) * p));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_controlArray, sizeof(int) * HTSZ));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hashTable, sizeof(int) * HTSZ));
        CubDebugExit(cudaMemcpy(d_controlArray, h_controlArray, sizeof(int) * HTSZ, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_hashTable, h_hashTable, sizeof(int) * HTSZ, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        CalculateHashPatternNew <<< 1, p >>> (d_patterns, d_controlArray, d_hashTable, d_patternHashes);
        cudaFree(d_patterns);
        cudaDeviceSynchronize();
        delete[] h_hashTable;
        delete[] h_controlArray;
        return { {d_controlArray, d_hashTable}, d_patternHashes};
    }
    static ull* Step3(char * d_data, ull* d_lookupTable) {
        ull* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(ull) * n));
        CalculateHashesNew << <n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaFree(d_data);
        cudaDeviceSynchronize();
        return d_a;
    }
    static ull* Step4(ull* d_a) {
        ull* d_prefixSum = NULL;
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_prefixSum, sizeof(ull) * n));
        cudaDeviceSynchronize();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumModMersennePrime, n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_prefixSum, sumModMersennePrime, n));
        cudaFree(d_a);
        cudaFree(d_temp_storage);
        return d_prefixSum;
    }
    static int* Step5(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
        int* h_output = new int [n];
        for (int i = 0; i < n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        FindMatchesNew << <n / 256, 256 >> > (d_prefixSum, d_lookupTable, d_controlArray, d_hashTable, d_output, d_patternHashes);
        CubDebugExit(cudaMemcpy(h_output, d_output, sizeof(int) * n, cudaMemcpyDeviceToHost));
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
        CopyDataToDevice();
        ull* d_lookupTable = Step1();
        static pair<pair<int*, int*>, ull*> p = Step2(test.d_patterns);
        ull* d_a = Step3(test.d_data, d_lookupTable);
        ull* d_prefixSum = Step4(d_a);
        int* h_output = Step5(d_prefixSum, d_lookupTable, p.first.first, p.first.second, p.second);
        return h_output;
    }
};
int main(){
    test.Validate(PaperImplementation::Execute());
    return 0;
}

