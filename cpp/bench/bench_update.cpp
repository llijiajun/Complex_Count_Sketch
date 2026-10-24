// Update-throughput benchmark: Count Sketch (real counters) vs Complex Count
// Sketch vs AMS (sample-count, scans width x AMS_num per update), same item
// stream, same machine.  Reports elements per second for each sketch.
#include <iostream>
#include <chrono>
#include <string>
#include "countsketch.hpp"

int main() {
    const long long N_ITEMS = 1000000;
    const double eps = 0.1, gam = 0.01;
    const int N_stream = 10000000;

    std::vector<std::string> items;
    items.reserve(N_ITEMS);
    for (long long i = 0; i < N_ITEMS; ++i)
        items.push_back(std::to_string(1000 + (i * 7919) % 10000));

    auto bench = [&](const char* name, auto make_and_run) {
        auto t0 = std::chrono::steady_clock::now();
        make_and_run();
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << name << ": " << (N_ITEMS / sec / 1e6) << " M items/s  ("
                  << sec << " s for " << N_ITEMS << " inserts)" << std::endl;
    };

    bench("CountSketch        ", [&] {
        sketch::CountSketch cs(eps, gam, 1);
        for (auto& s : items) cs.Insert(s.c_str(), (int)s.size());
    });
    bench("ComplexCountSketch ", [&] {
        sketch::ComplexCountSketch ccs(3, eps, gam, 1);
        for (auto& s : items) ccs.Insert(s.c_str(), (int)s.size());
    });
    bench("AMS (sample-count) ", [&] {
        sketch::AMS ams(eps, N_stream, gam, 1);
        for (auto& s : items) ams.Insert(s.c_str(), (int)s.size());
    });
    return 0;
}
