#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
// #include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define X86_MACHINEINSTR_PRINTER_PASS_NAME "Dummy X86 machineinstr printer pass"

namespace {

class X86MachineInstrPrinter : public MachineFunctionPass {
public:
    static char ID;

    X86MachineInstrPrinter() : MachineFunctionPass(ID) {
        initializeX86MachineInstrPrinterPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    StringRef getPassName() const override { return X86_MACHINEINSTR_PRINTER_PASS_NAME; }
};

char X86MachineInstrPrinter::ID = 0;

bool X86MachineInstrPrinter::runOnMachineFunction(MachineFunction &MF) {
    
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
INITIALIZE_PASS_BEGIN(X86MachineInstrPrinter, "x86-machineinstr-printer", X86_MACHINEINSTR_PRINTER_PASS_NAME, false, false)
INITIALIZE_PASS_END(X86MachineInstrPrinter, "x86-machineinstr-printer", X86_MACHINEINSTR_PRINTER_PASS_NAME, false, false)

namespace llvm {

FunctionPass *createX86MachineInstrPrinter() { return new X86MachineInstrPrinter(); }

}