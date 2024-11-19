// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//
#ifndef __IR2Vec_Utils__
#define __IR2Vec_Utils__

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <map>

namespace IR2Vec {
#define DIM 100
using Vector = llvm::SmallVector<double, DIM>;

void collectDataFromVocab(std::string vocab,
                          std::map<std::string, Vector> &opcMap);
void scaleVector(Vector &vec, float factor);
std::map<int, std::string> createOpcodeMap(llvm::Triple::ArchType archType);
std::map<std::string, std::string> createNotBrokenMap(llvm::Triple::ArchType archType);
} // namespace IR2Vec

#endif