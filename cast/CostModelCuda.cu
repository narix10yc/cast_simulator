#include "cast/CostModel.h"

using namespace cast;

// Definition of the dummy kernel
__global__ void emptyKernel() {}

// Definition of the member function
void CUDACostModel::measureLaunchOverheadOnce() {
    constexpr int N = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < N; ++i) {
        emptyKernel<<<1,1>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msElapsed = 0.0f;
    cudaEventElapsedTime(&msElapsed, start, stop);

    // Average overhead per kernel launch (seconds)
    measuredLaunchOverhead = (msElapsed / 1e3) / double(N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cerr << "Measured launch overhead: "
              << measuredLaunchOverhead << " s\n";
}
