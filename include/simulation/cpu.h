#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <map>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "simulation/irGen.h"
#include "qch/ast.h"

namespace simulation {

class CPUGenContext {
    std::map<uint32_t, std::string> gateMap;
    simulation::IRGenerator irGenerator;
    std::string fileName;
    std::error_code EC;
public:
    unsigned gateCount;
    std::stringstream shellStream;
    std::stringstream declStream;
    std::stringstream kernelStream;
    std::stringstream irStream;
    unsigned vecSizeInBits;
    RealTy realTy;
    unsigned nqubits;

    CPUGenContext(unsigned vecSizeInBits, std::string fileName)
        : fileName(fileName),
          vecSizeInBits(vecSizeInBits),
          realTy(RealTy::Double) {}
    
    void setRealTy(RealTy ty) { realTy = ty; }

    void logError(std::string msg) {}

    simulation::IRGenerator& getGenerator() { return irGenerator; }

    void generate(qch::ast::RootNode& root) {
        std::error_code EC;
        auto shellName = fileName + ".sh";
        auto shellFile = llvm::raw_fd_ostream(shellName, EC);
        std::cerr << "shell script will be written to: " << shellName << "\n";

        auto hName = fileName + ".h";
        auto hFile = llvm::raw_fd_ostream(hName, EC);
        std::cerr << "header file will be written to: " << hName << "\n";

        auto irName = fileName + ".ll";
        auto irFile = llvm::raw_fd_ostream(irName, EC);
        std::cerr << "IR file will be written to: " << irName << "\n";

        hFile << "#include <stdint.h>\n\n";
        if (realTy == RealTy::Double) {
            hFile << "typedef struct { double data[8]; } v8double;\n\n";
            kernelStream << "void simulate_circuit(double* real, double* imag) {\n";
        } else {
            hFile << "typedef struct { float data[8]; } v8float;\n\n";
            kernelStream << "void simulate_circuit(float* real, float* imag) {\n";
        }

        declStream << "extern \"C\" {\n";

        root.genCPU(*this);

        declStream << "}";
        kernelStream << "}";

        shellFile << shellStream.str();
        hFile << declStream.str() << "\n\n" << kernelStream.str();
        irGenerator.getModule().print(irFile, nullptr);

        shellFile.close();
        hFile.close();
        irFile.close();
    }
};

} // namespace simulation

#endif // SIMULATION_CPU_H_