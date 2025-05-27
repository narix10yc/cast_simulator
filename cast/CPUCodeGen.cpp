// #include "cast/cpu.h"
//
// #include "utils/iocolor.h"
// #include "utils/utils.h"
// #include "llvm/Bitcode/BitcodeWriter.h"
//
// #include <chrono>
// #include <fstream>
// #include <iomanip>
// #include <sstream>
//
// using namespace IOColor;
// using namespace cast;
// using namespace simulation;
// using namespace llvm;
//
// std::ostream& CodeGeneratorCPUConfig::display(int verbose,
//                                               std::ostream& os) const {
//   os << CYAN_FG << BOLD << "=== CodeGen Configuration ===\n" << RESET;
//
//   os << "Multi-threading " << ((multiThreaded) ? "enabled" : "disabled")
//      << ".\n";
//
//   if (installTimer)
//     os << "Timer installed\n";
//
//   os << "Dumping " << (writeRawIR ? "raw IR codes" : "bitcodes") << " to "
//      << (dumpIRToMultipleFiles ? "multiple files\n" : "a single file\n");
//
//   os << CYAN_FG << "Detailed IR settings:\n" << RESET;
//   irConfig.display(verbose, false, os);
//
//   os << CYAN_FG << BOLD << "=============================\n" << RESET;
//   return os;
// }
//
// void CodeGeneratorCPU::generate(const LegacyCircuitGraph& graph, int debugLevel,
//                                 bool forceInOrder) {
//   bool isSepKernel =
//       (config.irConfig.ampFormat == IRGeneratorConfig::SepFormat);
//
//   const std::string realTy =
//       (config.irConfig.precision == 32) ? "float" : "double";
//
//   std::stringstream externSS;
//   std::stringstream kernelSS;
//   std::stringstream metaDataSS;
//
//   externSS << "extern \"C\" {\n";
//
//   // simulation kernel declearation
//   kernelSS << "void simulation_kernel(";
//   if (isSepKernel)
//     kernelSS << realTy << "* re, " << realTy << "* im, ";
//   else
//     kernelSS << realTy << "* sv, ";
//
//   kernelSS << "const int nQubits";
//   if (config.multiThreaded)
//     kernelSS << ", const int nthreads";
//   kernelSS << ") {\n";
//
//   if (config.installTimer)
//     kernelSS << "  using clock = std::chrono::high_resolution_clock;\n"
//                 "  auto tic = clock::now();\n"
//                 "  auto tok = clock::now();\n";
//
//   if (config.multiThreaded)
//     kernelSS << " std::vector<std::thread> threads(nthreads);\n"
//              << " uint64_t chunkSize;\n";
//
//   kernelSS << "  uint64_t idxMax;\n";
//
//   // apply each gate kernel
//   kernelSS << "  for (const auto& data : _metaData) {\n"
//            << "    idxMax = 1ULL << (nQubits - data.nQubits - SIMD_S);\n";
//
//   std::string sv_arg = (isSepKernel) ? "re, im" : "sv";
//   if (config.multiThreaded)
//     kernelSS << "    chunkSize = idxMax / nthreads;\n"
//              << "    for (unsigned i = 0; i < nthreads; i++)\n"
//              << "      threads[i] = std::thread(data.func, " << sv_arg
//              << ", i*chunkSize, (i+1)*chunkSize, data.mPtr);\n"
//              << "    for (unsigned i = 0; i < nthreads; i++)\n"
//              << "      threads[i].join();\n";
//   else
//     kernelSS << "    data.func(" << sv_arg << ", 0, idxMax, data.mPtr);\n";
//
//   if (config.installTimer) {
//     kernelSS << "    tok = clock::now();\n"
//              << "    std::cerr << "
//                 "std::chrono::duration_cast<std::chrono::milliseconds>(tok - "
//                 "tic).count() << \" ms: \" << data.info << \"\\n\";\n"
//              << "    tic = clock::now();\n";
//   }
//
//   // meta data data type
//   metaDataSS << "struct _meta_data_t_ {\n"
//              << "  void (*func)(" << realTy << "*, ";
//   if (isSepKernel)
//     metaDataSS << realTy << "*, ";
//   metaDataSS << "uint64_t, uint64_t, const void*);\n"
//              << "  unsigned opCount;\n"
//              << "  unsigned nQubits;\n";
//   if (config.installTimer)
//     metaDataSS << "  const char* info;\n";
//   metaDataSS << "  const " << realTy << "* mPtr;\n"
//              << "};\n"
//              << "static const _meta_data_t_ _metaData[] = {\n";
//
//   auto allBlocks = graph.getAllBlocks();
//   if (forceInOrder)
//     std::ranges::sort(allBlocks, [](GateBlock* a, GateBlock* b) {
//       return a->id < b->id;
//     });
//
//   std::string kernelDir = fileName + "_kernels";
//   if (config.dumpIRToMultipleFiles) {
//     sys::fs::create_directory(kernelDir);
//   }
//
//   IRGenerator irGenerator(
//     config.irConfig, "mod_" + std::to_string(allBlocks[0]->id));
//   for (const auto& block : allBlocks) {
//     const auto& gate =* (block->quantumGate);
//     std::string kernelName = "kernel_block_" + std::to_string(block->id);
//     irGenerator.generateKernelDebug(gate, debugLevel, kernelName);
//     if (config.dumpIRToMultipleFiles) {
//       std::error_code ec;
//       llvm::raw_fd_ostream irFile(
//           kernelDir + "/kernel" + std::to_string(block->id) + ".ll", ec);
//       if (config.writeRawIR)
//         irGenerator.getModule()->print(irFile, nullptr);
//       else
//         WriteBitcodeToFile(*irGenerator.getModule(), irFile);
//       irFile.close();
//       irGenerator.~IRGenerator();
//       new (&irGenerator)
//           IRGenerator(config.irConfig, "mod_" + std::to_string(block->id));
//     }
//
//     externSS << " void " << kernelName << "(" << realTy << "*, ";
//     if (isSepKernel)
//       externSS << realTy << "*, ";
//     externSS << "uint64_t, uint64_t, const void*);\n";
//
//     // metaData
//     metaDataSS << " { "
//                << "&" << kernelName << ", " << block->quantumGate->opCount()
//                << ", " << block->nQubits() << ", ";
//
//     if (config.installTimer) {
//       std::stringstream infoSS;
//       infoSS << "block " << block->id << " ";
//       utils::printVector(block->getQubits(), infoSS);
//       metaDataSS << "\"" << infoSS.str() << "\", ";
//     }
//
//     metaDataSS << "(" << realTy << "[]){";
//     const auto* cMat = block->quantumGate->gateMatrix.getConstantMatrix();
//     assert(cMat);
//     for (const auto& elem : *cMat)
//       metaDataSS << std::setprecision(16) << elem.real() << ","
//                  << std::setprecision(16) << elem.imag() << ", ";
//     metaDataSS << "} },\n";
//   }
//
//   externSS << "};\n";
//   kernelSS << "  }\n}\n";
//   metaDataSS << "};\n";
//
//   std::ofstream hFile(fileName + ".h");
//   assert(hFile.is_open());
//
//   if (config.installTimer)
//     hFile << "#include <chrono>\n"
//              "#include <iostream>\n";
//
//   if (config.irConfig.precision == 32)
//     hFile << "#define USING_F32\n";
//   if (!isSepKernel)
//     hFile << "#define USING_ALT_KERNEL\n";
//
//   if (config.multiThreaded)
//     hFile << "#include <vector>\n"
//              "#include <thread>\n"
//              "#define MULTI_THREAD_SIMULATION_KERNEL\n\n";
//
//   hFile << "#include <cstdint>\n"
//         << "#define DEFAULT_nQubits " << graph.nQubits << "\n"
//         << "#define SIMD_S " << config.irConfig.simd_s << "\n"
//         << externSS.str() << "\n"
//         << metaDataSS.str() << "\n"
//         << kernelSS.str() << "\n";
//
//   hFile.close();
//
//   if (!config.dumpIRToMultipleFiles) {
//     std::error_code ec;
//     llvm::raw_fd_ostream irFile(fileName + ".ll", ec);
//     // irGenerator.getModule()->setModuleIdentifier(fileName + "_module");
//     // irGenerator.getModule()->setSourceFileName(fileName + ".ll");
//     if (config.writeRawIR)
//       irGenerator.getModule()->print(irFile, nullptr);
//     else
//       WriteBitcodeToFile(*irGenerator.getModule(), irFile);
//     irFile.close();
//   }
// }
//
// // generateCpuIrForRecompilation helper functions
// namespace {
//
// inline std::string getKernelName(GateBlock* block) {
//   return "kernel_" + std::to_string(block->id);
// }
//
// // write metadata to a C header file
// void writeMetadataHeaderFile(const std::vector<GateBlock*>& allBlocks,
//                              int nQubits, const std::string& fileName,
//                              const CodeGeneratorCPUConfig& config) {
//   bool isSepKernel =
//       (config.irConfig.ampFormat == IRGeneratorConfig::SepFormat);
//   const std::string realTy =
//       (config.irConfig.precision == 32) ? "float" : "double";
//   std::stringstream externSS;
//   std::stringstream kernelSS;
//   std::stringstream metaDataSS;
//
//   externSS << "extern \"C\" {\n";
//
//   // simulation kernel declaration
//   kernelSS << "void simulation_kernel(";
//   if (isSepKernel)
//     kernelSS << realTy << "* re, " << realTy << "* im, ";
//   else
//     kernelSS << realTy << "* sv, ";
//
//   kernelSS << "const int nQubits";
//   if (config.multiThreaded)
//     kernelSS << ", const int nthreads";
//   kernelSS << ") {\n";
//
//   if (config.installTimer)
//     kernelSS << "  using clock = std::chrono::high_resolution_clock;\n"
//                 "  auto tic = clock::now();\n"
//                 "  auto tok = clock::now();\n";
//
//   if (config.multiThreaded)
//     kernelSS << " std::vector<std::thread> threads(nthreads);\n"
//              << " uint64_t chunkSize;\n";
//
//   kernelSS << "  uint64_t idxMax;\n";
//
//   // apply each gate kernel
//   kernelSS << "  for (const auto& data : _metaData) {\n"
//            << "    idxMax = 1ULL << (nQubits - data.nQubits - SIMD_S);\n";
//
//   std::string sv_arg = (isSepKernel) ? "re, im" : "sv";
//   if (config.multiThreaded)
//     kernelSS << "    chunkSize = idxMax / nthreads;\n"
//              << "    for (unsigned i = 0; i < nthreads; i++)\n"
//              << "      threads[i] = std::thread(data.func, " << sv_arg
//              << ", i*chunkSize, (i+1)*chunkSize, data.mPtr);\n"
//              << "    for (unsigned i = 0; i < nthreads; i++)\n"
//              << "      threads[i].join();\n";
//   else
//     kernelSS << "    data.func(" << sv_arg << ", 0, idxMax, data.mPtr);\n";
//
//   if (config.installTimer) {
//     kernelSS << "    tok = clock::now();\n"
//              << "    std::cerr << "
//                 "std::chrono::duration_cast<std::chrono::milliseconds>(tok - "
//                 "tic).count() << \" ms: \" << data.info << \"\\n\";\n"
//              << "    tic = clock::now();\n";
//   }
//
//   // meta data data type
//   metaDataSS << "struct _meta_data_t_ {\n"
//              << "  void (*func)(" << realTy << "*, ";
//   if (isSepKernel)
//     metaDataSS << realTy << "*, ";
//   metaDataSS << "uint64_t, uint64_t, const void*);\n"
//              << "  unsigned opCount;\n"
//              << "  unsigned nQubits;\n";
//   if (config.installTimer)
//     metaDataSS << "  const char* info;\n";
//   metaDataSS << "  const " << realTy << "* mPtr;\n"
//              << "};\n"
//              << "static const _meta_data_t_ _metaData[] = {\n";
//
//   for (const auto& block : allBlocks) {
//     const auto& gate =* (block->quantumGate);
//     std::string kernelName = getKernelName(block);
//
//     externSS << " void " << kernelName << "(" << realTy << "*, ";
//     if (isSepKernel)
//       externSS << realTy << "*, ";
//     externSS << "uint64_t, uint64_t, const void*);\n";
//
//     // metaData
//     metaDataSS << " { "
//                << "&" << kernelName << ", " << block->quantumGate->opCount()
//                << ", " << block->nQubits() << ", ";
//
//     if (config.installTimer) {
//       std::stringstream infoSS;
//       infoSS << "block " << block->id << " ";
//       utils::printVector(block->getQubits(), infoSS);
//       metaDataSS << "\"" << infoSS.str() << "\", ";
//     }
//
//     metaDataSS << "(" << realTy << "[]){";
//     const auto* cMat = block->quantumGate->gateMatrix.getConstantMatrix();
//     assert(cMat);
//     for (const auto& elem : *cMat)
//       metaDataSS << std::setprecision(16) << elem.real() << ","
//                  << std::setprecision(16) << elem.imag() << ", ";
//     metaDataSS << "} },\n";
//   }
//
//   externSS << "};\n";
//   kernelSS << "  }\n}\n";
//   metaDataSS << "};\n";
//
//   std::ofstream hFile(fileName);
//   assert(hFile.is_open());
//
//   if (config.installTimer)
//     hFile << "#include <chrono>\n"
//              "#include <iostream>\n";
//
//   if (config.irConfig.precision == 32)
//     hFile << "#define USING_F32\n";
//   if (!isSepKernel)
//     hFile << "#define USING_ALT_KERNEL\n";
//
//   if (config.multiThreaded)
//     hFile << "#include <vector>\n"
//              "#include <thread>\n"
//              "#define MULTI_THREAD_SIMULATION_KERNEL\n\n";
//
//   hFile << "#include <cstdint>\n"
//         << "#define DEFAULT_nQubits " << nQubits << "\n"
//         << "#define SIMD_S " << config.irConfig.simd_s << "\n"
//         << externSS.str() << "\n"
//         << metaDataSS.str() << "\n"
//         << kernelSS.str() << "\n";
//
//   hFile.close();
// }
//
// using clock = std::chrono::high_resolution_clock;
//
// // write kernel ir to a llvm ir file named <dir>/kernels_t<threadIdx>.ll.
// // This file will contain (blockIdx1 - blockIdx0) kernels.
// // This function will own its own instance of IRGenerator
// void writeKernelsSingleIrFile(const std::vector<GateBlock*>& allBlocks,
//                               const std::string& dir,
//                               const CodeGeneratorCPUConfig& config,
//                               int blockIdx0, int blockIdx1, double* irGenTime) {
//
//   IRGenerator irGenerator(config.irConfig,
//                           "module_blocks" + std::to_string(blockIdx0) + "_to_" +
//                               std::to_string(blockIdx1));
//
//   auto tic = clock::now();
//   for (int i = blockIdx0; i < blockIdx1; i++) {
//     const auto& gate =* (allBlocks[i]->quantumGate);
//     std::string kernelName = getKernelName(allBlocks[i]);
//     irGenerator.generateKernel(gate, kernelName);
//   }
//   auto tok = clock::now();
//   std::chrono::duration<double> duration = tok - tic;
//   if (irGenTime) {
//    * irGenTime = duration.count();
//   }
//   std::error_code ec;
//   std::string filename = dir + "/kernel_" + std::to_string(blockIdx0) + "_to_" +
//                          std::to_string(blockIdx1) + ".ll";
//
//   llvm::raw_fd_ostream irFile(filename, ec);
//
//   if (config.writeRawIR)
//     irGenerator.getModule()->print(irFile, nullptr);
//   else
//     WriteBitcodeToFile(*irGenerator.getModule(), irFile);
//   irFile.close();
// }
//
// void writeKernelsMultipleIrFiles(const std::vector<GateBlock*>& allBlocks,
//                                  const std::string& dir,
//                                  const CodeGeneratorCPUConfig& config,
//                                  int blockIdx0, int blockIdx1,
//                                  double* irGenTime) {
//
//   IRGenerator irGenerator(config.irConfig,
//                           "module_block" +
//                               std::to_string(allBlocks[blockIdx0]->id));
//
//   auto tic = clock::now();
//   auto tok = clock::now();
//   double totalTime = 0.0;
//   std::chrono::duration<double> duration = tok - tic;
//
//   for (int i = blockIdx0; i < blockIdx1; i++) {
//     const auto& gate =* (allBlocks[i]->quantumGate);
//     std::string kernelName = getKernelName(allBlocks[i]);
//     tic = clock::now();
//     irGenerator.generateKernel(gate, kernelName);
//     tok = clock::now();
//     duration = tok - tic;
//     totalTime += duration.count();
//     std::error_code ec;
//     std::string filename =
//         dir + "/kernel_" + std::to_string(allBlocks[i]->id) + ".ll";
//
//     llvm::raw_fd_ostream irFile(filename, ec);
//
//     if (config.writeRawIR)
//       irGenerator.getModule()->print(irFile, nullptr);
//     else
//       WriteBitcodeToFile(*irGenerator.getModule(), irFile);
//     irFile.close();
//
//     if (i != blockIdx1 - 1) {
//       irGenerator.~IRGenerator();
//       new (&irGenerator)
//           IRGenerator(config.irConfig,
//                       "module_block" + std::to_string(allBlocks[i + 1]->id));
//     }
//   }
//   if (irGenTime) {
//    * irGenTime = totalTime;
//   }
//   return;
// }
//
// } // anonymous namespace
//
// void cast::generateCpuIrForRecompilation(const LegacyCircuitGraph& graph,
//                                          const std::string& dir,
//                                          const CodeGeneratorCPUConfig& config,
//                                          int nthreads) {
//
//   assert(nthreads > 0);
//   auto allBlocks = graph.getAllBlocks();
//   if (config.forceInOrder)
//     std::ranges::sort(allBlocks, [](GateBlock* a, GateBlock* b) {
//       return a->id < b->id;
//     });
//
//   sys::fs::create_directory(dir);
//   if (config.rmFilesInsideOutputDirectory) {
//     std::error_code ec;
//     for (sys::fs::directory_iterator file(dir, ec), end; file != end && !ec;
//          file.increment(ec)) {
//       if (sys::fs::is_regular_file(file->path())) {
//         std::error_code removeError = llvm::sys::fs::remove(file->path());
//         if (removeError) {
//           std::cerr << "Failed to remove file: " << file->path()
//                     << " Error: " << removeError.message() << "\n";
//         } else {
//           // std::cerr << "Removed file: " << file->path() << "\n";
//         }
//       }
//     }
//   }
//
//   if (nthreads == 1) {
//     // no multi-threading
//     writeMetadataHeaderFile(allBlocks, graph.nQubits,
//                             dir + "/kernel_metadata.h", config);
//     double irGenTime;
//     if (config.dumpIRToMultipleFiles)
//       writeKernelsMultipleIrFiles(allBlocks, dir, config, 0, allBlocks.size(),
//                                   &irGenTime);
//     else
//       writeKernelsSingleIrFile(allBlocks, dir, config, 0, allBlocks.size(),
//                                &irGenTime);
//     std::cerr << "IR Generation takes " << static_cast<int>(1e3 * irGenTime)
//               << " ms\n";
//     return;
//   }
//
//   // multi-threading
//   std::vector<std::thread> threads;
//   std::vector<double> irGenTimes(nthreads);
//   threads.reserve(nthreads);
//   int totalNBlocks = allBlocks.size();
//   int nBlocksPerThread = totalNBlocks / nthreads;
//
//   for (int threadIdx = 0; threadIdx < nthreads; threadIdx++) {
//     int blockIdx0 = threadIdx * nBlocksPerThread;
//     int blockIdx1 = (threadIdx == nthreads - 1)
//                         ? totalNBlocks
//                         : (threadIdx + 1) * nBlocksPerThread;
//     if (config.dumpIRToMultipleFiles)
//       threads.emplace_back(writeKernelsMultipleIrFiles, std::cref(allBlocks),
//                            dir, std::cref(config), blockIdx0, blockIdx1,
//                            irGenTimes.data() + threadIdx);
//     else
//       threads.emplace_back(writeKernelsSingleIrFile, std::cref(allBlocks), dir,
//                            std::cref(config), blockIdx0, blockIdx1,
//                            irGenTimes.data() + threadIdx);
//   }
//
//   // main thread writes metadata
//   writeMetadataHeaderFile(allBlocks, graph.nQubits, dir + "/kernel_metadata.h",
//                           config);
//
//   for (auto& t : threads)
//     t.join();
//
//   double irGenTime = 0.0;
//   for (const auto& t : irGenTimes) {
//     if (t > irGenTime)
//       irGenTime = t;
//   }
//   std::cerr << "IR Generation Time: " << static_cast<int>(1e3 * irGenTime)
//             << " ms\n";
//   return;
// }