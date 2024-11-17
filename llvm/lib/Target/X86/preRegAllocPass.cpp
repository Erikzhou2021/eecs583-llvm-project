#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
// #include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define PASS_NAME "Machine instruction printer after reg alloc"

namespace {

class PrintBeforeRegAlloc : public MachineFunctionPass {
public:
    static char ID;

    PrintBeforeRegAlloc() : MachineFunctionPass(ID) {
        initializePrintBeforeRegAllocPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    StringRef getPassName() const override { return PASS_NAME; }
};

char PrintBeforeRegAlloc::ID = 0;

bool PrintBeforeRegAlloc::runOnMachineFunction(MachineFunction &MF) {
    errs() << "Before \n --------------------------------------------- \n";
    for (auto &MBB : MF) {
        errs() << "Analyzing Basic Block: " << MBB.getName() << "\n";
        for (auto &MI : MBB) { // for each instruction in the basic block
            errs() << "  Instruction: " << MI << "\n";
            for (unsigned i = 0; i < MI.getNumOperands(); ++i) { // for each operand in the instruction
                MachineOperand &MO = MI.getOperand(i);
                // getReg() returns a Register object
                if (MO.isReg() && MO.getReg().isVirtual()) { // if operand is a virtual register
                    unsigned VirtReg = MO.getReg(); // get virtual register
                    // if (MRI.isVirtualRegister(VirtReg)) {
                    // unsigned PhysReg = MRI.getRegAllocHint(VirtReg).first; // get corresponding physical register
                    errs() << "    Virtual Register %" << VirtReg
                        << " has no mapping.\n";
                    // }
                }
                if(MO.isReg() && MO.getReg().isPhysical()){
                    unsigned PhysReg = MO.getReg();
                    errs() << " Physical Register %" << PhysReg << "\n";
                }
            }
        }
    }

    return false;
}

} // end of anonymous namespace

// INITIALIZE_PASS(X86MachineInstrPrinter, "x86-machineinstr-printer",
//     X86_MACHINEINSTR_PRINTER_PASS_NAME,
//     true, // is CFG only?
//     true  // is analysis?
// )
INITIALIZE_PASS_BEGIN(PrintBeforeRegAlloc, "x86-machineinstr-printer", PASS_NAME, true, true)
INITIALIZE_PASS_END(PrintBeforeRegAlloc, "x86-machineinstr-printer", PASS_NAME, true, true)

namespace llvm {

    FunctionPass *createPrintBeforeRegAlloc() { return new PrintBeforeRegAlloc(); }

}