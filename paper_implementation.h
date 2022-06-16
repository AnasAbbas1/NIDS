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
        delete[] h_hashTable;
        delete[] h_controlArray;
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        cudaFree(d_lookupTable);
        cudaFree(d_controlArray);
        cudaFree(d_hashTable);
        cudaDeviceSynchronize();
        return h_output;
    }
public:
    static int* Execute() {
        timer[0] = clock();
        //1.Load a preprocessed lookup table for di mod q (0 ≤ i ≤ q − 1)
        int* d_lookupTable = Step1();
        timer[0] = clock() - timer[0];
        timer[1] = clock();
        //2. Compute the values of h(Pk) for all k (0 ≤ k ≤ p − 1) in parallel and create the hash table HT using the calculated values.
        pair<int*, int*> p = Step2(g_d_patterns);
        timer[1] = clock() - timer[1];
        timer[2] = clock();
        //3.Compute the a0, a1,..., an−1 in parallel.
        int* d_a = Step3(g_d_data, d_lookupTable);
        timer[2] = clock() - timer[2];
        timer[3] = clock();
        //4.Compute the prefix-sums ˆa0, aˆ1,..., aˆn−1.
        int* d_prefixSum = Step4(d_a);
        timer[3] = clock() - timer[3];
        timer[4] = clock();
        //5.  For all j (0 ≤ j ≤ n − m), compute ( ˆaj+m−1 − aˆ j−1) · dm−n−j, which is equal to h(tjtj + 1 ... tj + m−1).If array control[h(tjtj + 1 ... tj + m−1)]  0 then compare the characters of text and pattern with Match(i, j).
        int* h_output = Step5(d_prefixSum, g_d_data, g_d_patterns, d_lookupTable, p.first, p.second);
        timer[4] = clock() - timer[4];
        return h_output;
    }
};
