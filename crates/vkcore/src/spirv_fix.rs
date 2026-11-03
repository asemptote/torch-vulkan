//! SPIR-V rewriting applied to every compiled kernel (by build.rs, and
//! checked by a test): `OpTypeForwardPointer`s are removed and the module's
//! types, constants and globals reordered so that each id is defined before
//! its first use. slang declares a forward pointer for every pointer to a
//! struct (e.g. a tensor of _int64.slang's I64), needed only for recursive
//! types, and Intel's Windows driver fails to build any pipeline with one
//! (returning success and no pipeline). Shared with build.rs through
//! `#[path]`, so it depends on nothing but `std`.

const OP_TYPE_FORWARD_POINTER: u32 = 39;
const OP_FUNCTION: u32 = 54;

/// Whether instruction `op` belongs to the types / constants / globals
/// section (where forward pointers matter).
fn is_global(op: u32) -> bool {
    matches!(op,
        19..=39          // OpTypeVoid .. OpTypeForwardPointer
        | 41..=52        // OpConstantTrue .. OpSpecConstantOp
        | 59             // OpVariable
        | 1              // OpUndef
        | 322            // OpTypePipeStorage
        | 4472           // OpTypeCooperativeMatrixKHR
    )
}

/// (result id, ids the instruction depends on) for a global instruction.
fn ids(op: u32, w: &[u32]) -> (Option<u32>, Vec<u32>) {
    match op {
        // types: result id first
        19..=22 | 26 | 27 | 31 | 322 => (Some(w[1]), vec![]), // void, bool, int, float, sampler, opaque
        23 | 24 => (Some(w[1]), vec![w[2]]),                    // vector, matrix: component type
        25 => (Some(w[1]), vec![w[2]]),                         // image: sampled type
        28 => (Some(w[1]), vec![w[2], w[3]]),                   // array: element type, length id
        29 => (Some(w[1]), vec![w[2]]),                         // runtime array
        30 | 33 => (Some(w[1]), w[2..].to_vec()),               // struct members, function return + params
        32 => (Some(w[1]), vec![w[3]]),                         // pointer: storage class, pointee
        4472 => (Some(w[1]), w[2..].to_vec()),
        // constants / globals: result type, result id
        41 | 42 | 46 | 1 => (Some(w[2]), vec![w[1]]),
        43 | 50 => (Some(w[2]), vec![w[1]]),                    // OpConstant, OpSpecConstant: literals
        48 | 49 => (Some(w[2]), vec![w[1]]),                    // OpSpecConstantTrue/False
        44 | 51 => (Some(w[2]), [&[w[1]], &w[3..]].concat()),   // composites
        52 => (Some(w[2]), [&[w[1]], &w[4..]].concat()),        // OpSpecConstantOp: opcode literal, then ids
        59 => (Some(w[2]), [&[w[1]], &w[4..]].concat()),        // OpVariable: storage class, initializer
        _ => (None, vec![]),
    }
}

/// The rewritten module, or None when it has no forward pointer (or its
/// globals could not be ordered without one: a recursive type).
pub fn drop_forward_pointers(words: &[u32]) -> Option<Vec<u32>> {
    if words.len() < 5 {
        return None;
    }
    // Split into instructions.
    let mut insts: Vec<&[u32]> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if n == 0 || i + n > words.len() {
            return None;
        }
        insts.push(&words[i..i + n]);
        i += n;
    }
    if !insts.iter().any(|w| w[0] & 0xffff == OP_TYPE_FORWARD_POINTER) {
        return None;
    }
    let first = insts.iter().position(|w| is_global(w[0] & 0xffff))?;
    let end = insts.iter().position(|w| w[0] & 0xffff == OP_FUNCTION).unwrap_or(insts.len());
    // The globals section, without its forward pointers.
    let globals: Vec<&[u32]> =
        insts[first..end].iter().copied().filter(|w| w[0] & 0xffff != OP_TYPE_FORWARD_POINTER).collect();
    let mut defined_at = std::collections::HashMap::new();
    let mut deps = Vec::with_capacity(globals.len());
    for (k, w) in globals.iter().enumerate() {
        let (result, d) = ids(w[0] & 0xffff, w);
        if let Some(r) = result {
            defined_at.insert(r, k);
        }
        deps.push(d);
    }
    // Depth-first in the original order: each instruction after what it uses.
    let mut order = Vec::with_capacity(globals.len());
    let mut state = vec![0u8; globals.len()]; // 0 new, 1 in progress, 2 done
    fn visit(k: usize, deps: &[Vec<u32>], at: &std::collections::HashMap<u32, usize>, state: &mut [u8], order: &mut Vec<usize>) -> bool {
        match state[k] {
            2 => return true,
            1 => return false, // a cycle: needs the forward pointer
            _ => {}
        }
        state[k] = 1;
        for d in &deps[k] {
            if let Some(&j) = at.get(d) {
                if j != k && !visit(j, deps, at, state, order) {
                    return false;
                }
            }
        }
        state[k] = 2;
        order.push(k);
        true
    }
    for k in 0..globals.len() {
        if !visit(k, &deps, &defined_at, &mut state, &mut order) {
            return None;
        }
    }
    let mut out = words[..5].to_vec();
    for w in &insts[..first] {
        out.extend_from_slice(w);
    }
    for k in order {
        out.extend_from_slice(globals[k]);
    }
    for w in &insts[end..] {
        out.extend_from_slice(w);
    }
    Some(out)
}

/// Whether the module still declares a forward pointer.
#[allow(dead_code)]  // (build.rs includes this file too)
pub fn has_forward_pointer(words: &[u32]) -> bool {
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if n == 0 {
            return false;
        }
        if words[i] & 0xffff == OP_TYPE_FORWARD_POINTER {
            return true;
        }
        i += n;
    }
    false
}
