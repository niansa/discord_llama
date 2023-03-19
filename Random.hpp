#ifndef _PHASMOENGINE_RANDOM_HPP
#define _PHASMOENGINE_RANDOM_HPP
#include <random>



class RandomGenerator {
    std::mt19937 rng;
    uint32_t initialSeed;

public:
    void seed() {
        rng.seed(initialSeed = std::random_device{}());
    }
    void seed(uint32_t customSeed) {
        rng.seed(initialSeed = customSeed);
    }

    unsigned getUInt() {
        std::uniform_int_distribution<unsigned> dist;
        return dist(rng);
    }
    unsigned getUInt(unsigned max) {
        std::uniform_int_distribution<unsigned> dist(0, max);
        return dist(rng);
    }
    unsigned getUInt(unsigned min, unsigned max) {
        std::uniform_int_distribution<unsigned> dist(min, max);
        return dist(rng);
    }
    double getDouble(double max) {
        std::uniform_real_distribution<double> dist(0.0, max);
        return dist(rng);
    }
    double getDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng);
    }
    bool getBool(float chance) {
        return getDouble(1.0) <= chance && chance != 0.0f;
    }
};
#endif
