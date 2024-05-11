#include "qch/ast.h"
#include "simulation/cpu.h"
#include "simulation/types.h"

using namespace qch::ast;
using namespace simulation;

void CircuitStmt::genCPU(CPUGenContext& ctx) const {
    ctx.nqubits = nqubits;
    for (auto& s : stmts)
        s->genCPU(ctx);
}

void GateApplyStmt::genCPU(CPUGenContext& ctx) const {
    uint64_t idxMax = 1ULL << (ctx.nqubits - ctx.vecSizeInBits - 1);
    if (name != "u3") {
        std::cerr << "skipped gate " << name << "\n";
        return;
    }

    // type string
    std::string typeStr =
        (ctx.getRealTy() == ir::RealTy::Double) ? "double*" : "float*";
    
    // get matrix
    auto mat = OptionalComplexMatrix2::FromEulerAngles(
                                parameters[0], parameters[1], parameters[2]);
    auto u3 = ir::U3Gate { static_cast<uint8_t>(qubits[0]), mat.ToIRMatrix(1e-8) };
    uint32_t u3ID = u3.getID();

    // get function name
    std::string funcName;
    if (auto it = ctx.u3GateMap.find(u3ID); it != ctx.u3GateMap.end()) {
        funcName = it->second;
    } else {
        // generate gate kernel
        auto func = ctx.getGenerator().genU3(u3);
        funcName = func->getName().str();
        ctx.u3GateMap[u3ID] = funcName;
        // generate func decl
        ctx.declStream << "void " << funcName << "(" << typeStr << ", ";
        if (ctx.getAmpFormat() == ir::AmpFormat::Separate)
            ctx.declStream << typeStr << ", ";
        ctx.declStream << "uint64_t, uint64_t, void*);\n";
    }
    
    // load u3 matrix
    ctx.kernelStream << std::setprecision(16) << "  u3m = {" 
        << mat.ar.value_or(0) << "," << mat.br.value_or(0) << ","
        << mat.cr.value_or(0) << "," << mat.dr.value_or(0) << ","
        << mat.ai.value_or(0) << "," << mat.bi.value_or(0) << ","
        << mat.ci.value_or(0) << "," << mat.di.value_or(0) << "};\n";
    
    // func call
    ctx.kernelStream << "  " << funcName << "(";
    if (ctx.getAmpFormat() == ir::AmpFormat::Separate)
        ctx.kernelStream << "real, imag";
    else
        ctx.kernelStream << "data";
    ctx.kernelStream << ", " << 0 << ", " << idxMax << ", u3m.data());\n";

}

