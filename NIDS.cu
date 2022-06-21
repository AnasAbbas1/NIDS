#include "constants.h"
#include "kernels.h"
#include "proposed_implementation.h"
#include "paper_implementation.h"
#include "testcase.h"
int main(){
    testcase::CopyDataToDevice();
    test.Validate(PaperImplementation::Execute(0), "paper", 0);
    test.Validate(ProposedImplementation::Execute(1), "proposed", 1);
    testcase::ClearDataFromDevice();
    cout << "proposed implementation is  " << timer[0][4] / (double)timer[1][4] << " times faster than paper implementation" << endl;
    return 0;
}
