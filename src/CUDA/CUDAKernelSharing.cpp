
// Attempt to implement kernel sharing for gates that meet the conditions
// static constexpr double SIMILARITY_THRESHOLD = 0.85;
// static constexpr unsigned MAX_GATES_PER_KERNEL = 4;
// static constexpr unsigned MAX_QUBITS_PER_GATE = 8;

// double qubitSimilarity(const QuantumGate& A, const QuantumGate& B) {
//   // Convert each qubit list to a std::set.
//   std::set<int> setA(A.qubits.begin(), A.qubits.end());
//   std::set<int> setB(B.qubits.begin(), B.qubits.end());

//   // Compute intersection
//   std::vector<int> interAB;
//   std::set_intersection(setA.begin(),
//                         setA.end(),
//                         setB.begin(),
//                         setB.end(),
//                         std::back_inserter(interAB));

//   // Compute union
//   std::vector<int> unionAB;
//   std::set_union(setA.begin(),
//                  setA.end(),
//                  setB.begin(),
//                  setB.end(),
//                  std::back_inserter(unionAB));

//   if (unionAB.empty()) {
//     return 0.0;
//   }
//   return double(interAB.size()) / double(unionAB.size());
// }

// static Function* getMultiKernelDeclaration(IRBuilder<>& B,
//                                            Module& M,
//                                            const std::string& funcName) {
//   FunctionType* fty = FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false);
//   auto* func = Function::Create(fty, Function::ExternalLinkage, funcName, M);

//   // Mark as a kernel
//   auto& ctx = M.getContext();
//   auto* mdString = MDString::get(ctx, "kernel");
//   auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
//   auto* kernelMetadata =
//       MDNode::get(ctx, {ValueAsMetadata::get(func), mdString, mdOne});
//   M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);

//   // Name the argument
//   func->getArg(0)->setName("p.sv");
//   return func;
// }

// /**
//  * Only group gates that pass threshold for qubitSimilarity,
//  * MAX_QUBITS_PER_GATE, and cap the group at MAX_GATES_PER_KERNEL.
//  */
// std::vector<std::vector<std::shared_ptr<QuantumGate>>>
// groupGatesByOverlapAndSize(
//     const std::vector<std::shared_ptr<QuantumGate>>& allGates) {
//   std::vector<std::vector<std::shared_ptr<QuantumGate>>> groups;
//   groups.reserve(allGates.size());

//   for (auto& g : allGates) {
//     // also skip if gate is too large
//     if (g->nQubits() > MAX_QUBITS_PER_GATE) {
//       // put it alone in its own group
//       groups.push_back({g});
//       continue;
//     }

//     bool placed = false;
//     for (auto& grp : groups) {
//       if (grp.size() >= MAX_GATES_PER_KERNEL)
//         continue; // group full
//       // check similarity vs group[0]
//       double sim = qubitSimilarity(*g, *grp.front());
//       if (sim >= SIMILARITY_THRESHOLD) {
//         grp.push_back(g);
//         placed = true;
//         break;
//       }
//     }
//     if (!placed) {
//       groups.emplace_back();
//       groups.back().push_back(g);
//     }
//   }

//   return groups;
// }

// CUDAKernelManager& CUDAKernelManager::genCUDAGateMulti(
//     const CUDAKernelGenConfig& config,
//     const std::vector<std::shared_ptr<QuantumGate>>& gateList,
//     const std::string& funcName) {
//   // 1) create new module
//   auto& cmp = createNewLLVMContextModulePair(funcName + "_Module");
//   IRBuilder<> B(*cmp.llvmContext);
//   Module& M = *cmp.llvmModule;

//   // 2) Build a function: “__global__ void kernel(double *pSv)”
//   FunctionType* fty = FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false);
//   Function* func = Function::Create(fty,
//                                     Function::ExternalLinkage,
//                                     funcName,
//                                     &M // place it in module M
//   );

//   // Mark as a kernel via metadata
//   {
//     auto& ctx = M.getContext();
//     auto* mdString = MDString::get(ctx, "kernel");
//     auto* mdOne = ConstantAsMetadata::get(B.getInt32(1));
//     auto* mdNode =
//         MDNode::get(ctx, {ValueAsMetadata::get(func), mdString, mdOne});
//     M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(mdNode);
//   }

//   // Name the single argument
//   Argument* pSvArg = func->getArg(0);
//   pSvArg->setName("p.sv");

//   // 3) Create the “entry” basic block and set insertion point
//   BasicBlock* entryBB = BasicBlock::Create(B.getContext(), "entry", func);
//   B.SetInsertPoint(entryBB);

//   // 4) Get global thread ID in 64-bit
//   Value* threadIdx64 = getGlobalTidCUDA(B); // e.g. 0..(some large)

//   // track total ops
//   size_t totalOps = 0;

//   // 5) For each gate, do:
//   for (auto& g : gateList) {
//     // (A) Create the bit-mask pointer. This replicates the single-gate
//     // “idxStartV” logic.
//     const auto& qubits = g->qubits;
//     int k = (int)qubits.size();
//     int highestQ = qubits.back();

//     // need the scalar type for the GEP
//     Type* scalarTy =
//         (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();

//     // Build idxStartV
//     Value* idxStartV = B.getInt64(0);
//     {
//       Value* tmpCounterV;
//       uint64_t mask = 0ULL;
//       int qIdx = 0;
//       int counterQ = 0;

//       // for each integer q in [0..highestQ], build partial mask
//       for (int q = 0; q <= highestQ; q++) {
//         if (q < qubits[qIdx]) {
//           // accumulate bits
//           mask |= (1ULL << (counterQ++));
//           continue;
//         }
//         // else q == qubits[qIdx]
//         ++qIdx;
//         if (mask != 0ULL) {
//           tmpCounterV =
//               B.CreateAnd(threadIdx64, B.getInt64(mask), "tmpCounter");
//           // shift by (qIdx - 1)
//           tmpCounterV =
//               B.CreateShl(tmpCounterV, B.getInt64(qIdx - 1), "tmpCounter");
//           idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
//           mask = 0ULL;
//         }
//         if (qIdx >= k) {
//           break; // done if we've processed all gate qubits
//         }
//       }

//       // Now handle bits above the last qubit
//       // mask = ~((1ULL << (gate->qubits.back() - k + 1)) - 1);
//       mask = ~((1ULL << (highestQ - k + 1)) - 1ULL);

//       tmpCounterV = B.CreateAnd(threadIdx64, B.getInt64(mask), "tmpCounter");
//       tmpCounterV = B.CreateShl(tmpCounterV, B.getInt64(k), "tmpCounter");
//       idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idxStart");
//     }

//     // multiply by 2 (for real+imag)
//     idxStartV = B.CreateShl(idxStartV, 1, "idxStart");

//     // Now do GEP
//     Value* maskedSvPtr =
//         B.CreateGEP(scalarTy, pSvArg, idxStartV, "maskedSvPtr");

//     // (B) Get matrix data
//     auto matData = getMatDataCUDA(B, g->gateMatrix, config);

//     // (C) Call genMatrixVectorMultiply_SharedTiled with this masked pointer
//     genMatrixVectorMultiply_SharedTiled(B,
//                                         config,
//                                         g->gateMatrix,
//                                         g->qubits,
//                                         matData,
//                                         maskedSvPtr, // use the bit-masked
//                                                      // pointer
//                                         scalarTy);

//     // track ops
//     totalOps += g->opCount(config.zeroTol);
//   }

//   // 6) return void
//   B.CreateRetVoid();

//   // 7) verify
//   if (verifyFunction(*func, &errs())) {
//     errs() << "[ERROR] multi function invalid.\n";
//   }

//   // 8) store kernel info
//   CUDAKernelInfo::CUDATuple emptyCT;
//   _cudaKernels.emplace_back(CUDAKernelInfo::PTXStringType(),
//                             config.precision,
//                             func->getName().str(),
//                             nullptr, // no single gate
//                             emptyCT,
//                             totalOps);

//   return *this;
// }

// CUDAKernelManager& CUDAKernelManager::genCUDAGatesFromCircuitGraphMulti(
//     const CUDAKernelGenConfig& config,
//     const CircuitGraph& graph,
//     const std::string& graphName) {
//   // gather all gates
//   std::vector<std::shared_ptr<QuantumGate>> allGates;
//   for (auto& block : graph.getAllBlocks()) {
//     allGates.push_back(block->quantumGate);
//   }

//   // group them
//   auto grouped = groupGatesByOverlapAndSize(allGates);

//   // for each group => build a single multi kernel
//   for (size_t i = 0; i < grouped.size(); i++) {
//     std::string fnName = graphName + "_multi_" + std::to_string(i);
//     genCUDAGateMulti(config, grouped[i], fnName);
//   }

//   return *this;
// }

// void CUDAKernelManager::dumpPTX(const std::string& kernelName,
//                                 llvm::raw_ostream& os) {
//   // First check if we have already compiled to PTX
//   for (auto& kernelInfo : standaloneKernels_) {
//     if (kernelInfo.llvmFuncName == kernelName) {
//       if (!kernelInfo.ptxString.empty()) {
//         os << "=== PTX for kernel '" << kernelName << "' ===\n";
//         os << kernelInfo.ptxString << "\n";
//         return;
//       }
//       break;
//     }
//   }

//   // If not found in compiled kernels, check the modules
//   for (auto& cmp : llvmContextModulePairs) {
//     auto& M = *cmp.llvmModule;
//     if (auto* F = M.getFunction(kernelName)) {
//       os << "=== Generating PTX for kernel '" << kernelName << "' ===\n";

//       // Initialize LLVM targets
//       llvm::InitializeAllTargetInfos();
//       llvm::InitializeAllTargets();
//       llvm::InitializeAllTargetMCs();
//       llvm::InitializeAllAsmPrinters();
//       llvm::InitializeAllAsmParsers();

//       // Configure for NVPTX
//       std::string error;
//       auto targetTriple = "nvptx64-nvidia-cuda";
//       auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
//       if (!target) {
//         os << "Error getting NVPTX target: " << error << "\n";
//         return;
//       }

//       // Set target options
//       llvm::TargetOptions opt;
//       auto RM = std::optional<llvm::Reloc::Model>();
//       auto targetMachine =
//           target->createTargetMachine(targetTriple, "sm_70", "+ptx60", opt, RM);

//       // Set up output stream
//       llvm::SmallString<0> ptxCode;
//       llvm::raw_svector_ostream ptxStream(ptxCode);

//       // Use legacy pass manager
//       llvm::legacy::PassManager pass;

//       // Version-agnostic file type selection
//       auto fileType =
// #if LLVM_VERSION_MAJOR >= 10
//           llvm::CodeGenFileType::AssemblyFile;
// #else
//           llvm::CGFT_AssemblyFile;
// #endif

//       if (targetMachine->addPassesToEmitFile(
//               pass, ptxStream, nullptr, fileType)) {
//         os << "Failed to generate PTX\n";
//         return;
//       }

//       pass.run(M);

//       os << ptxCode.str() << "\n";
//       return;
//     }
//   }

//   os << "No kernel found with name '" << kernelName << "'\n";
// }