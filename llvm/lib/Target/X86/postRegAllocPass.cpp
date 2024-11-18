#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
// #include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define PASS_NAME "Machine instruction printer after reg alloc"

namespace {

class PrintAfterRegAlloc : public MachineFunctionPass {
public:
    static char ID;

    PrintAfterRegAlloc() : MachineFunctionPass(ID) {
        initializePrintAfterRegAllocPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    StringRef getPassName() const override { return PASS_NAME; }
};

char PrintAfterRegAlloc::ID = 0;

bool PrintAfterRegAlloc::runOnMachineFunction(MachineFunction &MF) {
    errs() << "After \n --------------------------------------------- \n";
    for (auto &MBB : MF) {
        for (auto &MI : MBB) { // for each instruction in the basic block
            // errs() << "  Instruction: " << MI << "\n";
            errs() << MI.getOpcode() << " ";
            for (unsigned i = 0; i < MI.getNumOperands(); ++i) { // for each operand in the instruction
                MachineOperand &MO = MI.getOperand(i);
                if(MO.isReg() && MO.getReg().id() != 0){
                    errs() << MO.getReg() << " ";
                }
            }
            errs() << ", ";
        }
        errs() << "\n";
    }

    return false;
}

} // end of anonymous namespace

// INITIALIZE_PASS(X86MachineInstrPrinter, "x86-machineinstr-printer",
//     X86_MACHINEINSTR_PRINTER_PASS_NAME,
//     true, // is CFG only?
//     true  // is analysis?
// )
INITIALIZE_PASS_BEGIN(PrintAfterRegAlloc, "x86-machineinstr-printer", PASS_NAME, true, true)
INITIALIZE_PASS_END(PrintAfterRegAlloc, "x86-machineinstr-printer", PASS_NAME, true, true)

namespace llvm {

    FunctionPass *createprintAfterRegAlloc() { return new PrintAfterRegAlloc(); }

}