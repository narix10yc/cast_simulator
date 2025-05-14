#ifndef CAST_CPU_H
#define CAST_CPU_H

#include "simulation/ir_generator.h"

namespace simulation {
class IRGeneratorConfig;
}

namespace cast {
class LegacyCircuitGraph;
}
namespace cast {

struct CodeGeneratorCPUConfig {
  bool multiThreaded;
  bool installTimer;
  bool writeRawIR;
  bool dumpIRToMultipleFiles;
  bool rmFilesInsideOutputDirectory;
  bool forceInOrder; // force generating IR on ascending order of block ids
  simulation::IRGeneratorConfig irConfig;

  std::ostream& display(int verbose = 1, std::ostream& os = std::cerr) const;
};

class CodeGeneratorCPU {
private:
  std::string fileName;

public:
  CodeGeneratorCPU() : fileName(""), config() {}

  CodeGeneratorCPU(const CodeGeneratorCPUConfig& config,
                   const std::string& fileName = "gen_file")
      : fileName(fileName), config(config) {}

  CodeGeneratorCPUConfig config;

  /// @brief Generate IR
  /// @param forceInOrder: force generate IR according to block id
  void generate(const LegacyCircuitGraph& graph, int debugLevel = 0,
                bool forceInOrder = false);

  /// @brief Generate IR
  /// @param forceInOrder: force generate IR according to block id
  // void multiThreadGenerate(
  //         const CircuitGraph& graph, int nthreads,
  //         int debugLevel = 0, bool forceInOrder = false);

  std::ostream& displayConfig(int verbose = 1,
                              std::ostream& os = std::cerr) const {
    return config.display(verbose, os);
  }
};

void generateCpuIrForRecompilation(const LegacyCircuitGraph& graph,
                                   const std::string& dir,
                                   const CodeGeneratorCPUConfig& config,
                                   int nthreads = 1);

} // namespace cast

#endif // CAST_CPU_H