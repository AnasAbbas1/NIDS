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
        cudaDeviceSynchronize();
        timer[1][0] = clock();
        CubDebugExit(cudaMemcpy(ret, h_lookupTabe, sizeof(ull) * h_mps[5], cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        timer[1][0] = clock() - timer[1][0];
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
        timer[1][1] = clock();
        CalculateHashPatternNew <<< (h_p + 255) / 256 , min(256, h_p) >>> (d_patterns, d_controlArray, d_hashTable, d_patternHashes);
        cudaDeviceSynchronize();
        timer[1][1] = clock() - timer[1][1];
        delete[] h_hashTable;
        delete[] h_controlArray;
        cudaDeviceSynchronize();
        return { {d_controlArray, d_hashTable}, d_patternHashes};
    }
    static ull* Step3(char * d_data, ull* d_lookupTable) {
        ull* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(ull) * h_n));
        cudaDeviceSynchronize();
        timer[1][2] = clock();
        CalculateHashesNew << <h_n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaDeviceSynchronize();
        timer[1][2] = clock() - timer[1][2];
        cudaDeviceSynchronize();
        return d_a;
    }
    static ull* Step4(ull* d_a) {
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cudaDeviceSynchronize();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_a, sumModMersennePrime, h_n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        cudaDeviceSynchronize();
        timer[1][3] = clock();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_a, sumModMersennePrime,h_n));
        cudaDeviceSynchronize();
        timer[1][3] = clock() - timer[1][3];
        cudaFree(d_temp_storage);
        cudaDeviceSynchronize();
        return d_a;
    }
    static int* Step5(ull* d_prefixSum, ull* d_lookupTable, int* d_controlArray, int* d_hashTable, ull* d_patternHashes) {
        int* h_output = new int [h_n];
        for (int i = 0; i < h_n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * h_n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * h_n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        timer[1][4] = clock();
        FindMatchesNew << <h_n / 256, 256 >> > (d_prefixSum, d_lookupTable, d_controlArray, d_hashTable, d_output, d_patternHashes);
        cudaDeviceSynchronize();
        timer[1][4] = clock() - timer[1][4];
        CubDebugExit(cudaMemcpy(h_output, d_output, sizeof(int) * h_n, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        cudaFree(d_output);
        cudaFree(d_prefixSum);
        cudaFree(d_lookupTable);
        cudaFree(d_controlArray);
        cudaFree(d_hashTable);
        cudaFree(d_patternHashes);
        cudaDeviceSynchronize();
        return h_output;
    }
public:
    static int* Execute(int idx) {
        ull* d_lookupTable = Step1();
        static pair<pair<int*, int*>, ull*> p = Step2(g_d_patterns);
        ull* d_a = Step3(g_d_data, d_lookupTable);
        ull* d_prefixSum = Step4(d_a);
        int* h_output = Step5(d_prefixSum, d_lookupTable, p.first.first, p.first.second, p.second);
        return h_output;
    }
};
