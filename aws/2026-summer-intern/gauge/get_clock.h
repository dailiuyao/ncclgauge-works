#ifndef GET_CLOCK_H
#define GET_CLOCK_H

#include <stdint.h>  // for standard integer types

#ifdef __cplusplus
extern "C" {  // Ensure compatibility when calling from C++
#endif

typedef unsigned long long cycles_t;

#define MEASUREMENTS 200
#define USECSTART 10
#define USECSTEP 5

cycles_t get_cycles();  // Ensure this function is declared somewhere

double get_cpu_mhz();

#ifdef __cplusplus
}
#endif

#endif  // GET_CLOCK_H