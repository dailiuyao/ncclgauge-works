#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include "get_clock.h"

cycles_t get_cycles() {
    unsigned low, high;
    unsigned long long val;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    val = high;
    val = (val << 32) | low;
    return val;
}

double get_cpu_mhz() {
    struct timeval tv1, tv2;
    std::vector<long> x(MEASUREMENTS);
    std::vector<cycles_t> y(MEASUREMENTS);
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    double b, r_2;

    for (int i = 0; i < MEASUREMENTS; ++i) {
        cycles_t start = get_cycles();
        if (gettimeofday(&tv1, NULL)) {
            std::cerr << "gettimeofday failed." << std::endl;
            return 0;
        }
        do {
            if (gettimeofday(&tv2, NULL)) {
                std::cerr << "gettimeofday failed." << std::endl;
                return 0;
            }
        } while ((tv2.tv_sec - tv1.tv_sec) * 1000000 + (tv2.tv_usec - tv1.tv_usec) < USECSTART + i * USECSTEP);

        x[i] = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
        y[i] = get_cycles() - start;
    }

    for (int i = 0; i < MEASUREMENTS; ++i) {
        double tx = x[i], ty = y[i];
        sx += tx;
        sy += ty;
        sxx += tx * tx;
        syy += ty * ty;
        sxy += tx * ty;
    }

    b = (MEASUREMENTS * sxy - sx * sy) / (MEASUREMENTS * sxx - sx * sx);

    r_2 = (MEASUREMENTS * sxy - sx * sy) * (MEASUREMENTS * sxy - sx * sy) /
          ((MEASUREMENTS * sxx - sx * sx) * (MEASUREMENTS * syy - sy * sy));

    if (r_2 < 0.9) {
        std::cerr << "Correlation coefficient r^2: " << r_2 << " < 0.9" << std::endl;
        return 0;
    }
    return b;
}


