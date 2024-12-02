#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "Symbolic.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
// #include "llvm/Target/X86/X86RegisterInfo.h"

#include "json.hpp"
#include <sstream>
#include <string>

#include "llvm/Target/TargetMachine.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"

#include <set>

using namespace llvm;
using json = nlohmann::json;

#define PASS_NAME "Machine instruction printer after reg alloc"

namespace {

class IlpInstance : public MachineFunctionPass {
public:
    static char ID;

    IlpInstance() : MachineFunctionPass(ID) {
        initializeIlpInstancePass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    bool runOnMachineFunction(MachineFunction &MF) override;

    StringRef getPassName() const override { return PASS_NAME; }
};

char IlpInstance::ID = 0;

void IlpInstance::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(AU);
}

bool IlpInstance::runOnMachineFunction(MachineFunction &MF) {
    errs() << "Processing MachineFunction: " << MF.getName() << "\n";

    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    // MachineRegisterInfo &MRI = MF.getRegInfo();
    LiveIntervals &LIS = getAnalysis<LiveIntervals>();
    BitVector phys_regs = TRI->getAllocatableSet(MF);

    const TargetRegisterClass *GR32Class = TRI->getRegClass(X86::GR32RegClassID);
    const TargetRegisterClass *GR64Class = TRI->getRegClass(X86::GR64RegClassID);

    std::stringstream ss;
    json ilp_model;
    ilp_model["vars"] = json::array();
    ilp_model["objective"] = json::array();
    ilp_model["reg_assign"] = json::array();
    ilp_model["reg_move"] = json::array();

    for (MachineBasicBlock &MBB : MF) {
        std::set<Register> virt_regs;
        for (auto &MI : MBB) {
            for (unsigned i = 0; i < MI.getNumOperands(); ++i) { // for each operand in the instruction
                MachineOperand &MO = MI.getOperand(i);

                if (MO.isReg() && MO.getReg().isVirtual()) {
                    virt_regs.insert(MO.getReg());
                }
            }
        }
        
        for (Register vreg : virt_regs) {
            // objective function
            LiveInterval &LI = LIS.getInterval(vreg);

            // spill costs
            if (LIS.isLiveInToMBB(LI, &MBB)) {
                ss << "x_" << vreg << "_sig_" << MBB.getNumber();
                ilp_model["objective"].push_back({
                    {"cost", LI.weight()},
                    {"indicator", ss.str()}
                });
                ss.str(std::string());
            }

            // errs() << "REGCLASS " << MRI.getRegClass(vreg) << "\n";
            // std::stringstream ss;
            // ss << "x_" << vreg << "_" << << "_" << ;

            // move costs
            if (LIS.isLiveOutOfMBB(LI, &MBB)) {
                for (MachineBasicBlock *succ : MBB.successors()) {
                    for (const auto &reg : GR32Class->getRegisters()) {
                        ss << "x_" << vreg << "_" << reg << "_" << succ->getNumber();
                        ilp_model["objective"].push_back({
                            {"cost", 0},    // TODO: find suitable move cost
                            {"indicator", ss.str()}
                        });
                        ss.str(std::string());
                    }
                    for (const auto &reg : GR64Class->getRegisters()) {
                        ss << "x_" << vreg << "_" << reg << "_" << succ->getNumber();
                        ilp_model["objective"].push_back({
                            {"cost", 0},    // TODO: find suitable move cost
                            {"indicator", ss.str()}
                        });
                        ss.str(std::string());
                    }
                    ss << "x_" << vreg << "_sig_" << succ->getNumber();
                    ilp_model["objective"].push_back({
                        {"cost", 0},    // TODO: find suitable move cost
                        {"indicator", ss.str()}
                    });
                    ss.str(std::string());
                }
            }

            ilp_model["reg_assign"].push_back(json::array());
            for (const auto &reg : GR32Class->getRegisters()) {
                // constraint 7
                ss << "x_" << vreg << "_" << reg << "_" << MBB.getNumber();
                std::string xirn = ss.str();
                ilp_model["reg_assign"].back().push_back(xirn);
                ilp_model["var_list"].push_back(xirn);
                ss.str(std::string());

                // constraint 8 and 9
                for (MachineBasicBlock *succ : MBB.successors()) {
                    ss << vreg << "_" << reg << "_" << succ->getNumber();
                    ilp_model["var_list"].push_back("x_" + ss.str());
                    ilp_model["var_list"].push_back("c_" + ss.str());
                    ilp_model["reg_move"].push_back({
                        {"x1", "x_" + ss.str()},
                        {"x2", xirn},
                        {"c", "c_" + ss.str()}
                    });
                    ilp_model["reg_move"].push_back({
                        {"x1", xirn},
                        {"x2", "x_" + ss.str()},
                        {"c", "c_" + ss.str()}
                    });
                    ss.str(std::string());
                }

                
            }
            for (const auto &reg : GR64Class->getRegisters()) {
                // constraint 7
                ss << "x_" << vreg << "_" << reg << "_" << MBB.getNumber();
                std::string xirn = ss.str();
                ss.str(std::string());
                ilp_model["reg_assign"].back().push_back(xirn);
                ilp_model["var_list"].push_back(xirn);

                // constraint 8 and 9
                for (MachineBasicBlock *succ : MBB.successors()) {
                    ss << vreg << "_" << reg << "_" << succ->getNumber();
                    ilp_model["var_list"].push_back("x_" + ss.str());
                    ilp_model["var_list"].push_back("c_" + ss.str());
                    ilp_model["reg_move"].push_back({
                        {"x1", "x_" + ss.str()},
                        {"x2", xirn},
                        {"c", "c_" + ss.str()}
                    });
                    ilp_model["reg_move"].push_back({
                        {"x1", xirn},
                        {"x2", "x_" + ss.str()},
                        {"c", "c_" + ss.str()}
                    });
                    ss.str(std::string());
                }
            }
            // sigma
            // constraint 7
            ss << "x_" << vreg << "_sig_" << MBB.getNumber();
            std::string xirn = ss.str();
            ss.str(std::string());
            ilp_model["reg_assign"].back().push_back(xirn);
            ilp_model["var_list"].push_back(xirn);
            ss.str(std::string());

            // constraint 8 and 9
            for (MachineBasicBlock *succ : MBB.successors()) {
                ss << vreg << "_sig_" << succ->getNumber();
                ilp_model["var_list"].push_back("x_" + ss.str());
                ilp_model["var_list"].push_back("c_" + ss.str());
                ilp_model["reg_move"].push_back({
                    {"x1", "x_" + ss.str()},
                    {"x2", xirn},
                    {"c", "c_" + ss.str()}
                });
                ilp_model["reg_move"].push_back({
                    {"x1", xirn},
                    {"x2", "x_" + ss.str()},
                    {"c", "c_" + ss.str()}
                });
                ss.str(std::string());
            }
        }
    }

    // std::ofstream file("../../llvm/lib/Target/X86/ilp.json");
    std::ofstream file("../../ilp/ilp.json");
    if (file.is_open()) {
        file << ilp_model.dump(4);
        file.close();
    }

    return false;
}

} // end of anonymous namespace

// INITIALIZE_PASS(X86MachineInstrPrinter, "x86-machineinstr-printer",
//     X86_MACHINEINSTR_PRINTER_PASS_NAME,
//     true, // is CFG only?
//     true  // is analysis?
// )
INITIALIZE_PASS_BEGIN(IlpInstance, "x86-machineinstr-printer", PASS_NAME, true, true)
INITIALIZE_PASS_END(IlpInstance, "x86-machineinstr-printer", PASS_NAME, true, true)

namespace llvm {

    FunctionPass *createIlpInstance() { return new IlpInstance(); }

}