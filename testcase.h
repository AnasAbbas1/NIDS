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
        myfile.open(string(output_path + "data.txt").c_str());
        myfile << string(data);
        myfile.close();
    }
    void WritePatterns(char * patterns) {
        set<string>st;
        ofstream myfile;
        myfile.open(string(output_path + "patterns.txt").c_str());
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
        myfile.open((output_path + fileName).c_str());
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
    void SerialRabinKarp(){
        unordered_map<string, int> patIndex;
        unordered_map<int, bool> HashExist;
        set<pair<int, int> >st;
        int *lookupTable = new int [h_n];
        int *hashes = new int [h_n];
        for (int i = 0, current = 1; i < h_n; i++, current = (current * h_d) % h_q ) 
            lookupTable[i] = current;
        
        for(int i = 0; i < h_n; i++)
            hashes[i] = (( i ? hashes[i - 1] : 0) + (g_h_data[i] - 'a' + 1) * lookupTable[i]) % h_q;
        
        for (int i = 0; i < h_p; i++){
            string pattern = "";
            int patternHash = 0;
            for (int j = i * h_m; j < i * h_m + h_m; j++){
                pattern += g_h_patterns[j];
                patternHash = (patternHash + (g_h_patterns[j] - 'a' + 1) * lookupTable[j % h_m]) % h_q;
            }
            pattern += '\0';
            patIndex[pattern] = i;
            HashExist[patternHash] = true; 
        }
        for(int i = 0; i <= h_n - h_m; i++){
            int curHash = (hashes[i + h_m - 1] - (i ? hashes[i - 1] : 0) + h_q) % h_q;
            curHash = (curHash * lookupTable[(h_m + ((h_n - i + h_q - 2ll) / (h_q - 1ll)) * (h_q - 1ll) - h_n + i) % (h_q - 1ll)]) % h_q;

            if(HashExist[curHash]){
                string str = "";
                for (int j = i; j < i + h_m; j++)
                    str += g_h_data[j];
                str += '\0';
                if(patIndex.find(str) != patIndex.end()){
                    st.insert({{patIndex[str], i}});
                }
            }
        }
        while(st.size()){
            expectedMatches.push_back(*st.begin());
            st.erase(st.begin());
        }
    }
public:
    testcase() {
        cout << "generated data' size is " << h_n / (1 << 20) << "MB, expected to find  " << (h_p / (double)(1 << h_m)) * h_n << " pattern matches" << endl;
        GenerateInputData();
        WriteData(g_h_data);
        WritePatterns(g_h_patterns);
        clock_t start = clock();
        SerialRabinKarp();
        cout << "total execution time on cpu is " << (clock() - start) / (CLOCKS_PER_SEC / 1e3) << "ms" << endl;
        WriteMatches(expectedMatches, "Expected.txt");
    }
    static void CopyDataToDevice(){
        CubDebugExit(g_allocator.DeviceAllocate((void**)&g_d_data, sizeof(char) * h_n));
        CubDebugExit(cudaMemcpy(g_d_data, g_h_data, sizeof(char) * h_n, cudaMemcpyHostToDevice));
        CubDebugExit(g_allocator.DeviceAllocate((void**)&g_d_patterns, sizeof(char) * h_p * h_m));
        CubDebugExit(cudaMemcpy(g_d_patterns, g_h_patterns, sizeof(char) * h_p * h_m, cudaMemcpyHostToDevice));
        delete[] g_h_patterns;
        delete[] g_h_data;
    }
    static void ClearDataFromDevice(){
        cudaFree(g_d_data);
        cudaFree(g_d_patterns);
    }
    void Validate(int* h_output, string name) {
        vector<pair<int, int>>gpuMatches;
        for (int i = 0; i < h_n; i++) {
            if (h_output[i] != -1) {
                gpuMatches.push_back({h_output[i], i});
            }
        }
        sort(gpuMatches.begin(), gpuMatches.end());
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
            cout << name << " Code ran fine and output is as expected"<< endl;
            clock_t sum = 0;
            for(int i = 0; i < 5; i++){
                cout << "step #" << i + 1 << " : completed execution in " << timer[i] / (CLOCKS_PER_SEC / 1e6) << "us" << endl;
                sum += timer[i];
            }
            cout << "total execution time is " << sum / (CLOCKS_PER_SEC / 1e3) << "ms" << endl;
        }
        else {
            cout << "Results doesn't match, debug your code" << endl;
        }
        WriteMatches(gpuMatches, "Actual(" + name +").txt");
    }

}test;