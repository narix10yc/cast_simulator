#ifndef SAOT_CPU_H
#define SAOT_CPU_H

#include "simulation/ir_generator.h"

namespace simulation {
class IRGeneratorConfig;
}

namespace saot {
class CircuitGraph;
}
namespace saot {

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
  void generate(const CircuitGraph& graph, int debugLevel = 0,
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

void generateCpuIrForRecompilation(const CircuitGraph& graph,
                                   const std::string& dir,
                                   const CodeGeneratorCPUConfig& config,
                                   int nthreads = 1);

} // namespace saot

#endif // SAOT_CPU_H