#include "constants.h"
#include "kernels.h"
#include "proposed_implementation.h"
#include "paper_implementation.h"
#include "testcase.h"
int main(){
    testcase::CopyDataToDevice();
    test.Validate(ProposedImplementation::Execute(), "proposed");
    testcase::ClearDataFromDevice();
    return 0;
}
