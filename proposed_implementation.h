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
