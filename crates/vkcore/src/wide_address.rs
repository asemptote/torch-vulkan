//! Kernels that compute their addresses in 64 bits (docs/log/M56), for
//! dispatches that reach more than 4 GiB past an address they are given.
//!
//! The kernels address memory through buffer device addresses with 32-bit
//! indices: slang compiles `p[i]` and `p + i`, for a pointer `p` and an
//! `int` or `uint` `i`, to an `OpPtrAccessChain` whose element operand is
//! that 32-bit integer. On NVIDIA's driver (615.71, H100) the byte offset
//! wraps at 4 GiB: a float32 tensor's element 2^30 was read and written at
//! its element 0 (#142), and an index sign-extended to 64 bits first wraps
//! the same way. Mesa's drivers compute the address in 64 bits.
//!
//! [`integer_addresses`] rewrites each such access chain as integer
//! arithmetic, which every driver carries out in 64 bits:
//!
//! ```text
//! %u = OpConvertPtrToU %ulong %base
//! %i = OpSConvert %long %index        (OpUConvert %ulong for an unsigned index)
//! %o = OpIMul %ulong %i %stride       (the pointer type's ArrayStride)
//! %a = OpIAdd %ulong %u %o
//! %r = OpConvertUToPtr %type %a
//! ```
//!
//! It costs speed (the driver no longer sees a base and an offset: up to
//! 4x on NVIDIA for copies), so only dispatches that need it run the
//! rewritten kernel (lib.rs). Depends only on `std`.

use std::collections::{HashMap, HashSet};

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_MEMORY_MODEL: u32 = 14;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_PTR_ACCESS_CHAIN: u32 = 67;
const OP_DECORATE: u32 = 71;
const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
const OP_U_CONVERT: u32 = 113;
const OP_S_CONVERT: u32 = 114;
const OP_CONVERT_PTR_TO_U: u32 = 117;
const OP_CONVERT_U_TO_PTR: u32 = 120;
const OP_I_ADD: u32 = 128;
const OP_I_MUL: u32 = 132;
const CAP_INT64: u32 = 11;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const PHYSICAL_STORAGE_BUFFER: u32 = 5349;

/// Debug and annotation instructions, whose operands after the first are
/// literals or targets rather than (result type, result id).
fn is_annotation(op: u32) -> bool {
    matches!(op, 2..=8 | 71..=74 | 332 | 5632 | 5633)
}

fn inst(op: u32, operands: &[u32]) -> impl Iterator<Item = u32> + '_ {
    std::iter::once(((operands.len() as u32 + 1) << 16) | op).chain(operands.iter().copied())
}

/// What [`integer_addresses`] did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Rewritten {
    /// Access chains rewritten.
    pub chains: usize,
    /// Access chains on device addresses left as they were: their index's
    /// type or their pointer's stride was not found (none in the project's
    /// kernels: a test checks).
    pub unresolved: usize,
}

/// Rewrites every `OpPtrAccessChain` (and its in-bounds form) on a
/// `PhysicalStorageBuffer` pointer with a 32-bit index as 64-bit integer
/// arithmetic (above), adding `OpCapability Int64` and the 64-bit types and
/// constants it needs. Leaves a malformed module untouched.
pub fn integer_addresses(words: &mut Vec<u32>) -> Rewritten {
    let mut result = Rewritten::default();
    if words.len() < 5 {
        return result;
    }
    let mut insts: Vec<&[u32]> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if n == 0 || i + n > words.len() {
            return result;
        }
        insts.push(&words[i..i + n]);
        i += n;
    }

    // Integer types (id -> (width, signed)), device-address pointer types
    // and their strides, and the type of every id with an integer or
    // device-address pointer type.
    let mut ints: HashMap<u32, (u32, bool)> = HashMap::new();
    let mut psb: HashSet<u32> = HashSet::new();
    let mut stride_of: HashMap<u32, u32> = HashMap::new();
    let mut type_of: HashMap<u32, u32> = HashMap::new();
    let mut has_int64_cap = false;
    for w in &insts {
        let op = w[0] & 0xffff;
        match op {
            OP_CAPABILITY if w.len() == 2 && w[1] == CAP_INT64 => has_int64_cap = true,
            OP_DECORATE if w.len() == 4 && w[2] == DECORATION_ARRAY_STRIDE => {
                stride_of.insert(w[1], w[3]);
            }
            OP_TYPE_INT if w.len() == 4 => {
                ints.insert(w[1], (w[2], w[3] != 0));
            }
            OP_TYPE_POINTER if w.len() == 4 && w[2] == PHYSICAL_STORAGE_BUFFER => {
                psb.insert(w[1]);
            }
            _ => {
                // (an instruction with a result type has it first and the
                // result id second; no other instruction has a type's id
                // first, the annotations aside)
                if !is_annotation(op) && w.len() >= 3 && (ints.contains_key(&w[1]) || psb.contains(&w[1])) {
                    type_of.entry(w[2]).or_insert(w[1]);
                }
            }
        }
    }

    // The access chains to rewrite: (instruction, index (width, signed), stride).
    let mut chains: Vec<(usize, (u32, bool), u32)> = Vec::new();
    for (k, w) in insts.iter().enumerate() {
        let op = w[0] & 0xffff;
        if (op != OP_PTR_ACCESS_CHAIN && op != OP_IN_BOUNDS_PTR_ACCESS_CHAIN) || w.len() < 5 || !psb.contains(&w[1]) {
            continue;
        }
        let stride = type_of.get(&w[3]).and_then(|t| stride_of.get(t));
        match (type_of.get(&w[4]).and_then(|t| ints.get(t)), stride) {
            (Some(&index @ (32 | 64, _)), Some(&stride)) => chains.push((k, index, stride)),
            _ => result.unresolved += 1,
        }
    }
    if chains.is_empty() {
        return result;
    }

    // 64-bit types (new where the module has none) and stride constants.
    let mut bound = words[3];
    let mut fresh = || {
        bound += 1;
        bound - 1
    };
    let mut new_types: Vec<u32> = Vec::new();
    let mut int64 = |signed: bool, fresh: &mut dyn FnMut() -> u32| -> u32 {
        if let Some((&id, _)) = ints.iter().find(|(_, &(w, s))| w == 64 && s == signed) {
            return id;
        }
        let id = fresh();
        new_types.extend(inst(OP_TYPE_INT, &[id, 64, signed as u32]));
        id
    };
    let u64_type = int64(false, &mut fresh);
    let i64_type = if chains.iter().any(|c| c.1 == (32, true)) { int64(true, &mut fresh) } else { u64_type };
    let mut constants: Vec<u32> = Vec::new();
    let mut stride_ids: HashMap<u32, u32> = HashMap::new();
    for &(_, _, stride) in &chains {
        if let std::collections::hash_map::Entry::Vacant(e) = stride_ids.entry(stride) {
            let id = fresh();
            constants.extend(inst(OP_CONSTANT, &[u64_type, id, stride, 0]));
            e.insert(id);
        }
    }

    // New types before the first type (they depend on nothing), constants
    // before the first function, the capability after the last one.
    let first_type = insts.iter().position(|w| matches!(w[0] & 0xffff, 19..=39)).unwrap_or(insts.len());
    let first_function = insts.iter().position(|w| w[0] & 0xffff == OP_FUNCTION).unwrap_or(insts.len());
    let cap_at = insts.iter().rposition(|w| w[0] & 0xffff == OP_CAPABILITY).map_or_else(
        || insts.iter().position(|w| matches!(w[0] & 0xffff, OP_EXT_INST_IMPORT | OP_MEMORY_MODEL)).unwrap_or(0),
        |k| k + 1,
    );

    let mut out = Vec::with_capacity(words.len() + 20 * chains.len() + constants.len() + 12);
    out.extend_from_slice(&words[..5]);
    let mut next = chains.iter().peekable();
    for (k, w) in insts.iter().enumerate() {
        if k == cap_at && !has_int64_cap {
            out.extend(inst(OP_CAPABILITY, &[CAP_INT64]));
        }
        if k == first_type {
            out.extend_from_slice(&new_types);
        }
        if k == first_function {
            out.extend_from_slice(&constants);
        }
        let Some(&&(at, (width, signed), stride)) = next.peek() else {
            out.extend_from_slice(w);
            continue;
        };
        if at != k {
            out.extend_from_slice(w);
            continue;
        }
        next.next();
        let (base_type, base, index) = (type_of[&w[3]], w[3], w[4]);
        let u = fresh();
        out.extend(inst(OP_CONVERT_PTR_TO_U, &[u64_type, u, base]));
        let i = if width == 64 {
            index  // (OpIMul is the same mod 2^64 for either signedness)
        } else {
            let i = fresh();
            let convert = if signed { (OP_S_CONVERT, i64_type) } else { (OP_U_CONVERT, u64_type) };
            out.extend(inst(convert.0, &[convert.1, i, index]));
            i
        };
        let (o, a) = (fresh(), fresh());
        out.extend(inst(OP_I_MUL, &[u64_type, o, i, stride_ids[&stride]]));
        out.extend(inst(OP_I_ADD, &[u64_type, a, u, o]));
        if w.len() == 5 {
            out.extend(inst(OP_CONVERT_U_TO_PTR, &[w[1], w[2], a]));
        } else {
            // (further indices into the pointee: an access chain from the element)
            let p = fresh();
            out.extend(inst(OP_CONVERT_U_TO_PTR, &[base_type, p, a]));
            out.extend(inst(OP_ACCESS_CHAIN, &[&[w[1], w[2], p][..], &w[5..]].concat()));
        }
        result.chains += 1;
    }
    out[3] = bound;
    *words = out;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instructions(words: &[u32]) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let n = (words[i] >> 16) as usize;
            assert!(n > 0 && i + n <= words.len(), "malformed at word {i}");
            out.push(words[i..i + n].to_vec());
            i += n;
        }
        out
    }

    /// %2 = int, %3 = uint, %4 = PSB pointer to int (ArrayStride 4), %5 =
    /// int 7, %6 = uint 9, %7 = a pointer (undef), %8 and %9 = access chains
    /// with a signed and an unsigned index.
    fn module() -> Vec<u32> {
        vec![
            0x0723_0203, 0x0001_0500, 0, 10, 0,
            (2 << 16) | OP_CAPABILITY, 1,
            (2 << 16) | OP_CAPABILITY, 5347,  // PhysicalStorageBufferAddresses
            (3 << 16) | OP_MEMORY_MODEL, PHYSICAL_STORAGE_BUFFER, 1,
            (4 << 16) | OP_DECORATE, 4, DECORATION_ARRAY_STRIDE, 4,
            (4 << 16) | OP_TYPE_INT, 2, 32, 1,
            (4 << 16) | OP_TYPE_INT, 3, 32, 0,
            (4 << 16) | OP_TYPE_POINTER, 4, PHYSICAL_STORAGE_BUFFER, 2,
            (4 << 16) | OP_CONSTANT, 2, 5, 7,
            (4 << 16) | OP_CONSTANT, 3, 6, 9,
            (3 << 16) | 1, 4, 7,
            (5 << 16) | OP_FUNCTION, 0, 1, 0, 0,  // (only its position matters here)
            (5 << 16) | OP_PTR_ACCESS_CHAIN, 4, 8, 7, 5,
            (5 << 16) | OP_PTR_ACCESS_CHAIN, 4, 9, 7, 6,
        ]
    }

    #[test]
    fn rewrites_signed_and_unsigned_indices() {
        let mut w = module();
        assert_eq!(integer_addresses(&mut w), Rewritten { chains: 2, unresolved: 0 });
        let insts = instructions(&w);
        assert_eq!(insts[2], vec![(2 << 16) | OP_CAPABILITY, CAP_INT64]);
        // ulong = %10, long = %11 first in the types; the stride 4 as %12
        assert_eq!(insts[5], vec![(4 << 16) | OP_TYPE_INT, 10, 64, 0]);
        assert_eq!(insts[6], vec![(4 << 16) | OP_TYPE_INT, 11, 64, 1]);
        let at = insts.iter().position(|i| i[0] & 0xffff == OP_FUNCTION).unwrap();
        assert_eq!(insts[at - 1], vec![(5 << 16) | OP_CONSTANT, 10, 12, 4, 0]);
        assert_eq!(insts[at + 1..at + 6], [
            vec![(4 << 16) | OP_CONVERT_PTR_TO_U, 10, 13, 7],
            vec![(4 << 16) | OP_S_CONVERT, 11, 14, 5],
            vec![(5 << 16) | OP_I_MUL, 10, 15, 14, 12],
            vec![(5 << 16) | OP_I_ADD, 10, 16, 13, 15],
            vec![(4 << 16) | OP_CONVERT_U_TO_PTR, 4, 8, 16],
        ]);
        assert_eq!(insts[at + 7], vec![(4 << 16) | OP_U_CONVERT, 10, 18, 6]);
        assert_eq!(insts[at + 10], vec![(4 << 16) | OP_CONVERT_U_TO_PTR, 4, 9, 20]);
        assert_eq!(w[3], 21);
        // (once only: nothing left to rewrite)
        let before = w.clone();
        assert_eq!(integer_addresses(&mut w), Rewritten::default());
        assert_eq!(w, before);
    }

    /// Every kernel's device-address access chains are understood, and none
    /// is left afterwards.
    #[test]
    fn every_kernel() {
        let mut total = 0;
        for (name, spirv) in crate::kernels::SPIRV {
            let mut words = ash::util::read_spv(&mut std::io::Cursor::new(spirv)).unwrap();
            let r = integer_addresses(&mut words);
            assert_eq!(r.unresolved, 0, "{name}: access chains with an index or stride of unknown type");
            total += r.chains;
            let left = instructions(&words)
                .iter()
                .filter(|i| matches!(i[0] & 0xffff, OP_PTR_ACCESS_CHAIN | OP_IN_BOUNDS_PTR_ACCESS_CHAIN))
                .count();
            assert_eq!(left, 0, "{name}");
            if let Ok(dir) = std::env::var("VKCORE_WIDE_DUMP") {
                // (for spirv-val: VKCORE_WIDE_DUMP=dir cargo test every_kernel)
                let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/{name}.spv"), bytes).unwrap();
            }
        }
        assert!(total > 1000, "{total} access chains rewritten");
    }
}
