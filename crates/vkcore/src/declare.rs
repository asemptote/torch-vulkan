//! The memory a dispatch touches, for the stream's barriers (docs/log/M37).
//!
//! Kernels reach memory only through buffer device addresses in their push
//! constants: no kernel loads an address from memory. So every allocation a
//! dispatch can touch has an address in its push constants, and
//! [`push_accesses`] finds them by looking up every 8-byte word of the push
//! constants in the pool's live allocations. Each allocation found counts as
//! read and written in full, unless the caller declared, for that address,
//! the exact range the kernel reads or writes through it ([`Declared`]).

use crate::memory::{Located, Pool};
use crate::stream::Access;

/// The caller's declaration for one address in a dispatch's push constants:
/// the kernel touches memory through `ptr` only within `[lo, hi)`, and
/// writes it only if `write` is non-zero. (C layout: csrc/vk.h's VktRange.)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Declared {
    pub ptr: u64,
    pub lo: u64,
    pub hi: u64,
    pub write: u32,
    pub pad: u32,
}

/// The accesses of a dispatch with push constants `push`, or None if a word
/// points into the pool's memory outside every live allocation (a stale or
/// past-the-end address, or a value that merely looks like one): the
/// dispatch is then treated as touching all memory.
///
/// Every 8-byte-aligned word inside a live allocation is taken for an
/// address into it (a non-address that happens to fall there only adds a
/// conservative access). Its access is the caller's declarations for that
/// address, if it made any that hold it and lie within the allocation;
/// otherwise the whole allocation, read and written.
pub fn push_accesses(pool: &Pool, push: &[u8], declared: &[Declared]) -> Option<Vec<Access>> {
    let (span_lo, span_hi) = pool.span();
    let mut acc = Vec::with_capacity(8);
    for word in push.as_chunks::<8>().0 {
        let w = u64::from_le_bytes(*word);
        if w < span_lo || w >= span_hi {
            continue;
        }
        let (lo, hi) = match pool.locate(w) {
            Located::Allocation(lo, hi) => (lo, hi),
            Located::Unallocated => return None,
            Located::Outside => continue,
        };
        let mut covered = false;
        for d in declared.iter().filter(|d| d.ptr == w) {
            let holds = d.lo == d.hi || (d.lo <= w && w < d.hi);
            if holds && lo <= d.lo && d.hi <= hi {
                acc.push(Access { lo: d.lo, hi: d.hi, write: d.write != 0 });
                covered = true;
            }
        }
        if !covered {
            acc.push(Access { lo, hi, write: true });
        }
    }
    Some(acc)
}

/// A hash of the bytes of `[lo, lo + len)` (host memory at `host`) outside
/// the ranges `skip` (GPU addresses, sorted by `lo` and merged).
///
/// # Safety
/// `host` must be valid for reading `len` bytes.
pub unsafe fn hash_outside(lo: u64, len: u64, host: *const u8, skip: &[(u64, u64)]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut mix = |bytes: &[u8]| {
        let (words, rest) = bytes.as_chunks::<8>();
        for c in words {
            h = (h ^ u64::from_le_bytes(*c)).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
        }
        for &b in rest {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
        }
        h = h.wrapping_add(bytes.len() as u64);
    };
    let hi = lo + len;
    let mut at = lo;
    for &(s_lo, s_hi) in skip {
        let (s_lo, s_hi) = (s_lo.clamp(lo, hi), s_hi.clamp(lo, hi));
        if s_lo > at {
            mix(std::slice::from_raw_parts(host.add((at - lo) as usize), (s_lo - at) as usize));
        }
        at = at.max(s_hi);
    }
    if hi > at {
        mix(std::slice::from_raw_parts(host.add((at - lo) as usize), (hi - at) as usize));
    }
    h
}

/// `ranges` sorted by start and merged where they overlap or touch.
pub fn merged(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (lo, hi) in ranges {
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflicts() {
        let r = |lo, hi| Access { lo, hi, write: false };
        let w = |lo, hi| Access { lo, hi, write: true };
        assert!(!r(0, 10).conflicts(&r(5, 15))); // read after read
        assert!(w(0, 10).conflicts(&r(5, 15))); // RAW / WAR
        assert!(r(5, 15).conflicts(&w(0, 10)));
        assert!(w(0, 10).conflicts(&w(9, 10))); // WAW
        assert!(!w(0, 10).conflicts(&w(10, 20))); // adjacent
        assert!(!w(0, 10).conflicts(&r(10, 20)));
    }

    #[test]
    fn merging_and_hashing() {
        assert_eq!(merged(vec![(5, 8), (0, 2), (2, 3), (7, 10)]), vec![(0, 3), (5, 10)]);
        let mut bytes = [1u8; 64];
        let h = |b: &[u8; 64], skip: &[(u64, u64)]| unsafe { hash_outside(1000, 64, b.as_ptr(), skip) };
        let before = h(&bytes, &[(1010, 1020)]);
        bytes[12] = 7; // inside the skipped range
        assert_eq!(h(&bytes, &[(1010, 1020)]), before);
        bytes[20] = 7; // just past it
        assert_ne!(h(&bytes, &[(1010, 1020)]), before);
    }
}
