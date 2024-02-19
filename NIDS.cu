#include "constants.h"
#include "kernels.h"
#include "proposed_implementation.h"
#include "paper_implementation.h"
#include "testcase.h"
int main(int argc, char *argv[]){
    SetConstants(argc, argv);
    CopyConstants();
    testcase test;
    testcase::CopyDataToDevice();
    test.Validate(PaperImplementation::Execute(0), "paper", 0);
    test.Validate(ProposedImplementation::Execute(1), "proposed", 1);
    testcase::ClearDataFromDevice();
    FinalReport();
    return 0;
}
