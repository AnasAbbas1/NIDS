struct CustomSum
{
    CUB_RUNTIME_FUNCTION __host__ __device__ __forceinline__
        usi operator()(const usi& a, const unsigned int& b) const {
        return paper_prime > a + b ? a + b : a + b - paper_prime;
    }
}sumMod;
struct CustomSumNew
{
    CUB_RUNTIME_FUNCTION __host__ __device__ __forceinline__
        ull operator()(const ull& a, const ull& b) const {
            ull ans = 0;
            for(int j = d_masksz - 1; j >= 0; j--){
                unsigned int sum = ((a & d_masks[j]) >> d_cumShifts[j]) + ((b & d_masks[j]) >> d_cumShifts[j]);
                sum = sum >= d_mps[j] ? sum - d_mps[j] : sum;
                ans |= sum;
                ans <<= d_oppShifts[j];
            }
        return ans;
    }
}sumModMersennePrime;
__global__ void CalculateHashPattern(char* d_patterns, int* d_controlArray, int* d_hashTable) {
    int patternIndex = blockIdx.x * blockDim.x + threadIdx.x, patternHash = 0;

    for (int i = patternIndex * d_m; i < patternIndex * d_m + d_m; i++)
        patternHash = (patternHash * d_prime + (d_patterns[i] - 'a' + 1)) % paper_prime;

    while (atomicAdd(&d_controlArray[patternHash], 1) != 0)
        patternHash = (patternHash + 1) % paper_prime;

    d_hashTable[patternHash] = patternIndex;
}
__global__ void CalculateHashPatternNew(char* d_patterns, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
    int patternIndex = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void CalculateHashes(usi* d_a, char* d_data, usi* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = (d_lookupTable[(d_n - i - 1) % (paper_prime_1)] * (d_data[i] - ascii_code_1)) % paper_prime;
}
__global__ void CalculateHashesNew(ull* d_a, char* d_data, ull* d_lookupTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[i] = 0;
    unsigned int ascii = (d_data[i] - ascii_code_1);
    unsigned int power = (d_n - i - 1);
    ull lookUpValue = d_lookupTable[power % four_primes_1];
    for(int j = d_masksz - 1; j >= 0; j--){
        unsigned int exactLookUpValue = ((j > 3 ? d_lookupTable[power % d_mps_1[j]] : lookUpValue) & d_masks[j]) >> d_cumShifts[j];
        unsigned int hash = exactLookUpValue * ascii; //hash =  (ds[j]^(i mod (mps[j] - 1)) % mps[j]) * data[i]
        hash = (hash & d_mps[j]) + (hash >> d_shifts[j]);
        hash = hash >= d_mps[j] ? hash - d_mps[j] : hash;
        d_a[i] |= hash;
        d_a[i] <<= d_oppShifts[j];
    }
    
}
__global__ void FindMatches(usi* d_prefixSum, char* d_data, char* d_patterns, usi* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= d_n - d_m) {
        usi hash = d_prefixSum[j + d_m - 1];
        usi prevHash = j ? d_prefixSum[j - 1] : 0;
        hash = hash > prevHash ? hash - prevHash : (paper_prime - prevHash + hash); 
        unsigned int lookUpValue = d_lookupTable[(d_m - d_n + j + d_paper_c) % (paper_prime_1)];
        hash = (hash * lookUpValue) % paper_prime;
        while (d_controlArray[hash]) {
            usi patternIndex = d_hashTable[hash];
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
            hash = (hash == paper_prime_1 ? 0 : hash + 1);
        }
    }
}
__global__ void FindMatchesNew(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, int* d_output, ull* d_patternHashes) {
    ull j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + d_m - 1 < d_n) {
        ull hash = 0;
        ull curHash = d_prefixSum[j + d_m - 1];
        ull prevHash = j ? d_prefixSum[j - 1] : 0;
        ull power = d_proposed_c + d_m - d_n + j;
        ull lookUpValue = d_lookupTable[power % four_primes_1];
        for(int k = d_masksz - 1; k >= 0; k--){
            ull tmp = (curHash & d_masks[k]) >> d_cumShifts[k];
            tmp = (tmp + d_mps[k]) - ((prevHash & d_masks[k]) >> d_cumShifts[k]);
            tmp = (tmp >= d_mps[k] ? tmp - d_mps[k] : tmp);
            tmp *= (( k > 3 ? d_lookupTable[power % d_mps_1[k]] : lookUpValue) & d_masks[k]) >> d_cumShifts[k] ;
            tmp = (tmp & d_mps[k]) + (tmp >> d_shifts[k]);
            hash |= (tmp >= d_mps[k] ? tmp - d_mps[k] : tmp);
            hash <<= d_oppShifts[k];
        }

        unsigned int hashOfHash = hash & d_HTMSK;
        while (d_controlArray[hashOfHash]) {
            if (hash == d_patternHashes[d_hashTable[hashOfHash]]) {
                d_output[j] = d_hashTable[hashOfHash];
                return;
            }
            hashOfHash = (hashOfHash == d_HTMSK) ? 0: hashOfHash + 1;
        }
    }
}
