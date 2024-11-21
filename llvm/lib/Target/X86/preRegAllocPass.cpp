#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "Symbolic.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"

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
    const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    for (auto &MBB : MF) {
        errs() << MBB.getNumber() + 3002 << "$";
        for (auto &MI : MBB) { // for each instruction in the basic block
            // errs() << "  Instruction: " << MI << "\n";
            errs() << TII->getName(MI.getOpcode()) << "|";
            errs() << MI.getOpcode() << ":";
            bool printed = false;
            for (unsigned i = 0; i < MI.getNumOperands(); ++i) { // for each operand in the instruction
                MachineOperand &MO = MI.getOperand(i);
                if(MI.isBranch() && MO.isMBB()){
                    if(printed){
                        errs() << ",";
                    }
                    errs() << MO.getMBB()->getNumber() + 3002;
                    printed = true;
                }
                if(MO.isReg() && MO.getReg().id() != 0){
                    if(printed){
                        errs() << ",";
                    }
                    if(MO.getReg().isVirtual()){
                        errs() << MO.getReg().virtRegIndex() + 1000;
                    }
                    else{
                        errs() << MO.getReg();
                    }
                    printed = true;
                }
            }
            errs() << " ";
        }
        errs() << "\n";
    }

    // const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    // for (auto &MBB : MF) {
    //     for (auto &MI : MBB) {
    //         errs() << TII->getName(MI.getOpcode()) << "\n";
    //     }
    // }
    // MIR2Vec_Symbolic* symbolic = new MIR2Vec_Symbolic("/home/erzh/llvm-project/llvm/lib/Target/X86/vocabulary/seedEmbedding_5500E_100D.txt");
    // symbolic->generateSymbolicEncodings(MF);
    // auto instVecMap = symbolic->getInstVecMap();

    return false;
}

// bool PrintBeforeRegAlloc::runOnMachineFunction(MachineFunction &MF) {
//     errs() << "Before \n --------------------------------------------- \n";
//     for (auto &MBB : MF) {
//         // errs() << "Analyzing Basic Block: " << MBB.getName() << "\n";
//         for (auto &MI : MBB) { // for each instruction in the basic block
//             // errs() << "  Instruction: " << MI << "\n";
//             errs() << MI.getOpcode() << " ";
//             for (unsigned i = 0; i < MI.getNumOperands(); ++i) { // for each operand in the instruction
//                 MachineOperand &MO = MI.getOperand(i);
//                 if(MO.isReg() && MO.getReg().id() != 0){
//                     if(MO.getReg().isVirtual()){
//                         errs() << MO.getReg().virtRegIndex() << " ";
//                     }
//                     else{
//                         errs() << MO.getReg() << " ";
//                     }
//                 }
//             }
//             errs() << ", ";
//         }
//         errs() << "\n";
//     }

//     return false;
// }

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