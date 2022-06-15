#include "constants.h"
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
    ull hashOfHash = patternHash & d_HTMSK;
    while (atomicAdd(&d_controlArray[hashOfHash], 1) != 0)
        hashOfHash = (hashOfHash == d_HTMSK) ? 0: hashOfHash + 1;

    d_hashTable[hashOfHash] = patternIndex;
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

        ull hashOfHash = hash & d_HTMSK;
        while (d_controlArray[hashOfHash]) {
            if (hash == d_patternHashes[d_hashTable[hashOfHash]]) {
                d_output[j] = d_hashTable[hashOfHash];
                return;
            }
            hashOfHash = (hashOfHash == d_HTMSK) ? 0: hashOfHash + 1;
        }
    }
}
