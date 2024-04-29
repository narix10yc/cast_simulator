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

    auto u3 = U3Gate::FromAngles(qubits[0],
        parameters[0], parameters[1], parameters[2]);

    std::stringstream funcNameSS;
    funcNameSS << "u3_" << ((ctx.realTy == RealTy::Double) ? "f64_" : "f32_")
        << ctx.gateCount << "_" 
        << std::setfill('0') << std::setw(8) << std::hex << u3.getID();

    std::string funcName = funcNameSS.str();

    ctx.getGenerator().genU3(u3, funcName, ctx.realTy);

    ctx.declStream << "void " << funcName;
    if (ctx.realTy == RealTy::Double)
        ctx.declStream << "(double*, double*, uint64_t, uint64_t, v8double);\n";
    else
        ctx.declStream << "(float*, float*, uint64_t, uint64_t, v8float);\n";

    ctx.kernelStream << "  " << funcName << "(real, imag, 0, " << idxMax << ",\n    "
        << ((ctx.realTy == RealTy::Double) ? "(v8double){" : "(v8float){")
        << std::setprecision(16)
        << u3.mat.ar.value_or(0) << "," << u3.mat.br.value_or(0) << ","
        << u3.mat.cr.value_or(0) << "," << u3.mat.dr.value_or(0) << ","
        << u3.mat.ai.value_or(0) << "," << u3.mat.bi.value_or(0) << ","
        << u3.mat.ci.value_or(0) << "," << u3.mat.di.value_or(0) << "});\n";

    ctx.gateCount ++;
}

