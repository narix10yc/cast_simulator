#ifndef SIMULATION_CODEGEN_H_
#define SIMULATION_CODEGEN_H_

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include <vector>
#include <array>

#include "quench/QuantumGate.h"

namespace simulation {
struct IRGeneratorConfig {
    enum class AmpFormat { Alt, Sep };

    int simd_s                      = 2;
    int precision                   = 64;
    AmpFormat ampFormat             = AmpFormat::Alt;   
    bool useFMA                     = true;
    bool useFMS                     = true;
    // parallel bits deposite from BMI2
    bool usePDEP                    = true;
    bool loadMatrixInEntry          = true;
    bool loadVectorMatrix           = false;
    bool forceDenseKernel           = false;
    double zeroSkipThres            = 1e-8;
    double shareMatrixElemThres     = 0.0;
    bool shareMatrixElemUseImmValue = false;

    bool checkConfliction(std::ostream& os) const;

    std::ostream&
    display(int verbose = 1, bool title = true, std::ostream& os = std::cerr) const;
};

/// @brief IR Generator.
class IRGenerator {
    llvm::LLVMContext llvmContext;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::Module> mod;
    IRGeneratorConfig _config;

    using AmpFormat = IRGeneratorConfig::AmpFormat;
public:
    IRGenerator(const std::string& moduleName = "myModule") : 
        llvmContext(),
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>(moduleName, llvmContext)),
        _config() {}

    IRGenerator(const IRGeneratorConfig& irConfig, const std::string& moduleName = "myModule") : 
        llvmContext(),
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>(moduleName, llvmContext)),
        _config(irConfig) {}

    const llvm::Module& getModule() const { return *mod; }
    llvm::Module& getModule() { return *mod; }

    llvm::IRBuilder<>& getBuilder() { return builder; }

    IRGeneratorConfig& config() { return _config; }
    const IRGeneratorConfig& config() const { return _config; }

    void loadFromFile(const std::string& fileName);
    
    /// @brief Generate the IR that applies new_aa = aa + bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag special values are +1, -1, or 0
    /// @return aa + bb * cc. Possible nullptr, when aa is nullptr and bbFlag = 0
    llvm::Value* genMulAdd(
            llvm::Value* aa, llvm::Value* bb, llvm::Value* cc, int bbFlag,
            const llvm::Twine& bbccName = "", const llvm::Twine& aaName = "");

    /// @brief Generate the IR that applies new_aa = aa - bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to -bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag special values are +1, -1, or 0
    /// @return aa - bb * cc
    llvm::Value* genMulSub(
            llvm::Value* aa, llvm::Value* bb, llvm::Value* cc, int bbFlag,
            const llvm::Twine& bbccName = "", const llvm::Twine& aaName = "");

    llvm::Function*
    generateKernel(
            const quench::quantum_gate::QuantumGate& gate,
            const std::string& funcName = "") {
        return generateKernelDebug(gate, 0, funcName);
    }

    llvm::Function*
    generateKernelDebug(const quench::quantum_gate::QuantumGate& gate,
                        int debugLevel,
                        const std::string& funcName = "");

};


} // namespace simulation

#endif // SIMULATION_CODEGEN_H_