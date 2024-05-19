#include <cstdint>
#include <array>

extern "C" {
void f64_s2_sep_u2q_k1l0(double*, double*, uint64_t, uint64_t, void*);
}

void simulate_circuit(double *real, double *imag, uint64_t, uint64_t, void*) {
  std::array<double, 8> u3m;
  std::array<double, 32> u2qm;
  u2qm = {0.737507562384852,-0.2137762451627083,-0.06874037038864651,-0.1174565628410863,-0.1174565628410862,-0.0687403703886465,-0.2137762451627083,0.737507562384852,-0.06874037038864646,-0.1174565628410863,0.737507562384852,-0.2137762451627083,-0.2137762451627083,0.737507562384852,-0.1174565628410862,-0.06874037038864647,0.04837723707316119,-0.1668968325343431,-0.5190338368417354,-0.3037597672342311,-0.3037597672342311,-0.5190338368417354,-0.1668968325343431,0.0483772370731612,-0.5190338368417354,-0.3037597672342311,0.04837723707316119,-0.1668968325343431,-0.1668968325343431,0.04837723707316118,-0.3037597672342311,-0.5190338368417354};
  f64_s2_sep_u2q_k1l0(real, imag, 0, 4611686018427387904, u3m.data());
}