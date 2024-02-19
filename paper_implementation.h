class PaperImplementation{
private:
    static usi* Step1() {
        usi* ret = NULL;
        usi *h_lookupTable = new usi [paper_prime];
        for (int i = 0, current = 1; i < paper_prime; i++, current = (current * h_prime) % paper_prime ) 
            h_lookupTable[i] = current;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&ret, sizeof(usi) * paper_prime));
        cudaDeviceSynchronize();
        timer[0][0] = clock();
        CubDebugExit(cudaMemcpy(ret, h_lookupTable, sizeof(usi) * paper_prime, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        timer[0][0] = clock() - timer[0][0];
        delete[] h_lookupTable;
        return ret;
    }
    static pair<int*, int*> Step2(char * d_patterns) {
        int* d_controlArray = NULL, * d_hashTable = NULL, * h_controlArray = new int[paper_prime], * h_hashTable = new int[paper_prime];
        for (int i = 0; i < paper_prime; i++) {
            h_controlArray[i] = 0;
            h_hashTable[i] = -1;
        }
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_controlArray, sizeof(int) * paper_prime));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hashTable, sizeof(int) * paper_prime));
        CubDebugExit(cudaMemcpy(d_controlArray, h_controlArray, sizeof(int) * paper_prime, cudaMemcpyHostToDevice));
        CubDebugExit(cudaMemcpy(d_hashTable, h_hashTable, sizeof(int) * paper_prime, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        timer[0][1] = clock();
        CalculateHashPattern <<< (h_p + 255) / 256 , min(256, h_p) >>> (d_patterns, d_controlArray, d_hashTable);
        cudaDeviceSynchronize();
        timer[0][1] = clock() - timer[0][1];
        delete[] h_hashTable;
        delete[] h_controlArray;
        cudaDeviceSynchronize();
        return { d_controlArray, d_hashTable };
    }
    static usi * Step3(char * d_data, usi* d_lookupTable) {
        usi* d_a = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_a, sizeof(usi) * h_n));
        cudaDeviceSynchronize();
        timer[0][2] = clock();
        CalculateHashes << <h_n / 256, 256 >> > (d_a, d_data, d_lookupTable);
        cudaDeviceSynchronize();
        timer[0][2] = clock() - timer[0][2];
        return d_a;
    }
    static usi* Step4(usi* d_a) {
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_a, sumMod, h_n));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        cudaDeviceSynchronize();
        timer[0][3] = clock();
        CubDebugExit(DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_a, d_a, sumMod, h_n));
        cudaDeviceSynchronize();
        timer[0][3] = clock() - timer[0][3];
        cudaFree(d_temp_storage);
        return d_a;
    }
    static int* Step5(usi* d_prefixSum, char* d_data, char* d_patterns, usi* d_lookupTable, int* d_controlArray, int* d_hashTable) {
        int* h_output = new int [h_n];
        for (int i = 0; i < h_n; i++)
            h_output[i] = -1;
        int* d_output = NULL;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * h_n));
        CubDebugExit(cudaMemcpy(d_output, h_output, sizeof(int) * h_n, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        timer[0][4] = clock();
        FindMatches << <h_n / 256, 256 >> > (d_prefixSum, d_data, d_patterns, d_lookupTable, d_controlArray, d_hashTable, d_output);
        cudaDeviceSynchronize();
        timer[0][4] = clock() - timer[0][4];
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
    static int* Execute(int _idx) {
        //1.Load a preprocessed lookup table for di mod q (0 ≤ i ≤ q − 1)
        usi* d_lookupTable = Step1();
        //2. Compute the values of h(Pk) for all k (0 ≤ k ≤ p − 1) in parallel and create the hash table HT using the calculated values.
        pair<int*, int*> p = Step2(g_d_patterns);
        //3.Compute the a0, a1,..., an−1 in parallel.
        usi* d_a = Step3(g_d_data, d_lookupTable);
        //4.Compute the prefix-sums ˆa0, aˆ1,..., aˆn−1.
        usi* d_prefixSum = Step4(d_a);
        //5.  For all j (0 ≤ j ≤ n − m), compute ( ˆaj+m−1 − aˆ j−1) · dm−n−j, which is equal to h(tjtj + 1 ... tj + m−1).If array control[h(tjtj + 1 ... tj + m−1)]  0 then compare the characters of text and pattern with Match(i, j).
        int* h_output = Step5(d_prefixSum, g_d_data, g_d_patterns, d_lookupTable, p.first, p.second);
        return h_output;
    }
};
