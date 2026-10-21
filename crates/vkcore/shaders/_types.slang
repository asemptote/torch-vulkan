// Derives PARAM_T (and PARAM_DOUBLE) from the kernel's T axis: float64 and
// complex128 kernels take float parameters as double, everything else as
// float. Included by _strided.slang and _strided4.slang.

#pragma once

#if defined(T_f64) || defined(T_c128)
#define PARAM_T double
#define PARAM_DOUBLE 1
#else
#define PARAM_T float
#endif
