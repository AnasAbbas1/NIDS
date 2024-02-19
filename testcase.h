struct testcase {
private:
    string input_str;
    vector<pair<int, int>>expectedMatches;
    char* StringGeneration(int sz) {
        char* ret = new char[sz + 1];
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
            outputFile << "Duplicate pattern occurred" << endl; 
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
    vector<string>patternPool;
    void span(int idx, string cur){
        if(idx == h_m){
            patternPool.push_back(cur);
        }else{
            for(int ch = 0; ch < h_d; ch++){
                span(idx + 1, cur + (char)('a' + ch));
            }
        }
    }
    char * PatternsGeneration(){
        set<string>st;
        long long allValidPatterns = 1;
        for(int i = 0; i < h_m; i++){
            allValidPatterns *= h_d;
            allValidPatterns = min((double)allValidPatterns, 1e9);
        }
        if(h_p / (double)allValidPatterns <= 0.5){
            while(st.size() != h_p){
                char * ptrn = StringGeneration(h_m);
                st.insert(string(ptrn));
            }
        }else{
            span(0, "");
            for(int i = 0; i < h_p; i++){
                st.insert(patternPool[i]);
            }
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
        vector<int> *matchPos = new vector<int>[h_p];
        vector<int> * hashPatterns = new vector<int>[paper_prime];
        int *hashes = new int [h_n];
        unsigned int *lookupTable = new unsigned int [paper_prime];
        for (int i = 0, current = 1; i < paper_prime; i++, current = (current * h_d) % paper_prime ) 
            lookupTable[i] = current;
        
        for(int i = 0; i < h_n; i++){
            hashes[i] = (lookupTable[(h_n - i - 1) % (paper_prime_1)] * (g_h_data[i] - 'a' + 1)) % paper_prime;
            if(i)
                hashes[i] = (hashes[i] + hashes[i - 1]) % paper_prime;
        }
        
        for (int i = 0; i < h_p; i++){
            string pattern = "";
            int patternHash = 0;
            for (int j = i * h_m; j < i * h_m + h_m; j++){
                pattern += g_h_patterns[j];
                patternHash = (patternHash * h_d + g_h_patterns[j] - 'a' + 1) % paper_prime;
            }
            pattern += '\0';
            hashPatterns[patternHash].push_back(i);
        }

        for(int i = 0; i <= h_n - h_m; i++){
            unsigned int curHash = (hashes[i + h_m - 1] - (i ? hashes[i - 1] : 0) + paper_prime) % paper_prime;
            curHash = (curHash * lookupTable[(h_m - h_n + i + d_paper_c) % (paper_prime_1)]) % paper_prime;
            for(int j = 0; j < hashPatterns[curHash].size(); j++){
                bool match = true;
                int curPatIndex = hashPatterns[curHash][j];
                for(int k = 0, idx = curPatIndex * h_m; k < h_m; k++){
                    if (g_h_patterns[idx + k] != g_h_data[i + k]){
                        match = false;
                        break;
                    }
                }
                if(match){
                    matchPos[curPatIndex].push_back(i);
                    break;
                }
            }
        }
        for(int i = 0; i < h_p; i++){
            for(int j = 0; j < matchPos[i].size(); j++){
                expectedMatches.push_back({i, matchPos[i][j]});
            }
        }
    }
public:
    testcase() {
        outputFile << "generated data' size is " << h_n / (1 << 20) << "MB" << endl;
        GenerateInputData();
        if(write_data)
            WriteData(g_h_data);
        if(write_data)
            WritePatterns(g_h_patterns);
        cpu = clock();
        SerialRabinKarp();
        cpu =  clock() - cpu;
        outputFile << "total execution time on cpu is " << cpu / (CLOCKS_PER_SEC / 1e3) << "ms" << endl;
        if(write_data)
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
    void Validate(int* h_output, string name, int idx) {
        vector<pair<int, int>>gpuMatches;
        for (int i = 0; i < h_n; i++) {
            if (h_output[i] != -1) {
                gpuMatches.push_back({h_output[i], i});
            }
        }
        sort(gpuMatches.begin(), gpuMatches.end());
        correctResult = true;
        if (gpuMatches.size() != expectedMatches.size()) {
            outputFile << "sizes are not equal expected size is " << expectedMatches.size() << " and actual size is " << gpuMatches.size() << endl;
            correctResult = false;
        }
        for (int i = 0, limit = 0;i < min(gpuMatches.size(), expectedMatches.size()); i++) {
            if (gpuMatches[i].first != expectedMatches[i].first || gpuMatches[i].second != expectedMatches[i].second) {
                outputFile << "Mismatch at position: " << i << endl;
                limit++;
                correctResult = false;
                if(limit >= 100){
                    break;
                }
            }
        }
        if (correctResult) {
            outputFile << name << " Code ran fine and output is as expected"<< endl;
            clock_t sum = 0;
            for(int i = 0; i < 5; i++){
                outputFile << "step #" << i + 1 << " : completed execution in " << timer[idx][i] / (CLOCKS_PER_SEC / 1e6) << "us" << endl;
                sum += timer[idx][i];
            }
            outputFile << "total execution time is " << sum / (CLOCKS_PER_SEC / 1e3) << "ms" << endl;
        }
        else {
            outputFile << "Results doesn't match, debug your code" << endl;
        }
        if(write_data)
            WriteMatches(gpuMatches, "Actual(" + name +").txt");
    }

};