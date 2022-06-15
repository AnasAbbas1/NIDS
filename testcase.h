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