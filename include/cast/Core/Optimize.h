#ifndef CAST_CORE_OPTIMIZE_H
#define CAST_CORE_OPTIMIZE_H

#include "cast/Core/IRNode.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/FusionConfig.h"
#include "utils/utils.h"

namespace cast {

class Optimizer {
private:
  struct PassBase {
    virtual ~PassBase() = default;
    virtual void run(ir::CircuitNode& circuit) = 0;
  };
  template<typename PassType>
  struct PassWrapper : public PassBase {
    PassType pass;
    PassWrapper(PassType&& p) : pass(std::move(p)) {}
    void run(ir::CircuitNode& circuit) override { pass(circuit); }
  };

  struct PassItem {
    std::string name;
    std::unique_ptr<PassBase> pass;
    PassItem(const std::string& n, std::unique_ptr<PassBase> p)
      : name(n), pass(std::move(p)) {}
  };

  std::vector<PassItem> passes;
public:
  template<typename PassType>
  void addPass(const std::string& name, PassType&& pass) {
    using WrapperType = PassWrapper<std::decay_t<PassType>>;
    passes.emplace_back(
      name,
      std::make_unique<WrapperType>(std::forward<PassType>(pass))
    );
  }

  void run(ir::CircuitNode& circuit, int verbose=0) {
    if (verbose <= 0) {
      for (const auto& [name, pass] : passes)
        pass->run(circuit);
      return;
    }
    for (const auto& [name, pass] : passes) {
      utils::timedExecute(
        [&circuit, &pass]() { pass->run(circuit); },
        name.c_str()
      );
    }

  }
}; // class Optimizer


/// @brief Optimize the circuit by trying to fuse away all single-qubit gates,
/// including those in if statements.
void applyCanonicalizationPass(ir::CircuitNode& circuit, double swapTol);

void applyGateFusionPass(ir::CircuitNode& circuit, const FusionConfig* config);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H