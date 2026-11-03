// A raw 64-bit kernel parameter slot, packed by the host per kernel type (a
// double's bits for float64 and complex128 kernels, a float's bits in the low
// word for other floating kernels, an int64 for integer ones), held as two
// 32-bit words (lo, hi). A kernel then needs the Int64 capability only if it
// reads the whole slot as a 64-bit integer (int64 kernels), and Float64 only
// as a double: float32, int32 and 16-bit kernels run on GPUs without either
// (Mali GPUs have neither).

#pragma once

typedef uint2 slot64;

float slot_float(slot64 s) { return asfloat(s.x); }
int slot_int(slot64 s) { return asint(s.x); }  // (an int64's low word: the value mod 2^32)
double slot_double(slot64 s) { return asdouble(s.x, s.y); }
int64_t slot_int64(slot64 s) { return int64_t(uint64_t(s.x) | (uint64_t(s.y) << 32)); }

// The integer scalar type of integer kernels: int64 for int64 kernels, int
// otherwise (whose results are the int64 computation mod 2^32 anyway).
#if defined(T_i64)
typedef int64_t slot_int_t;
slot_int_t slot_integer(slot64 s) { return slot_int64(s); }
#else
typedef int slot_int_t;
slot_int_t slot_integer(slot64 s) { return slot_int(s); }
#endif
