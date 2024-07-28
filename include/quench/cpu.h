#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

namespace quench::cpu {

struct CodeGeneratorCPUConfig {
    int simd_s;
    int precision;
    bool multiThreaded;
    bool installTimer;
    int overrideNqubits;
    bool loadMatrixInEntry;
    bool loadVectorMatrix;
    bool usePDEP; // parallel bit deposite
    bool dumpIRToMultipleFiles;
    bool enablePrefetch;

    std::ostream& display(std::ostream& os = std::cerr) const {
        os << Color::CYAN_FG << "== CodeGen Configuration ==\n" << Color::RESET
           << "SIMD s:      " << simd_s << "\n"
           << "Precision:   " << "f" << precision << "\n";
        
        os << "Multi-threading "
           << ((multiThreaded) ? "enabled" : "disabled")
           << ".\n";
        
        if (installTimer)
            os << "Timer installed\n";
        if (overrideNqubits > 0)
            os << "Override nqubits = " << overrideNqubits << "\n";
        
        os << "Detailed IR settings:\n"
           << "  load matrix in entry: " << ((loadMatrixInEntry) ? "true" : "false") << "\n"
           << "  load vector matrix:   " << ((loadVectorMatrix) ? "true" : "false") << "\n"
           << "  use PDEP:             " << ((usePDEP) ? "true" : "false") << "\n";

        os << Color::CYAN_FG << "===========================\n" << Color::RESET;
        return os;
    }

};

class CodeGeneratorCPU {
private:
    std::string fileName;
public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : fileName(fileName), 
          config({.simd_s=1, .precision=64, .multiThreaded=false,
                  .installTimer=false,
                  .overrideNqubits=-1,
                  .loadMatrixInEntry=true,
                  .loadVectorMatrix=true,
                  .usePDEP=true,
                  .dumpIRToMultipleFiles=false,
                  .enablePrefetch=false}) {}

    CodeGeneratorCPUConfig config;

    void generate(const circuit_graph::CircuitGraph& graph, int verbose=0);

    std::ostream& displayConfig(std::ostream& os = std::cerr) const {
        return config.display(os);
    }
    
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H