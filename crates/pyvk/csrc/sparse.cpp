// Sparse COO tensors on device "vk" (M18): a sparse tensor's indices
// [sparse_dim, nnz] (int64) and values [nnz, *dense dims] are ordinary "vk"
// tensors, so most of PyTorch's sparse implementation (written in terms of
// tensor ops on those two) serves as is, registered on SparsePrivateUse1:
// construction, the getters, resizing, copies, clone, conversions to dense,
// and the zero-preserving unary functions applied to the values. The kernels
// that PyTorch writes per device (CPU loops, CUDA kernels) are here in tensor
// ops, which run as Vulkan kernels:
//   _coalesce      indices linearized to one int64 key per entry, a stable
//                  sort, runs of equal keys summed into their first entry in
//                  index order (as the CPU's loop) by index_add
//   add            dense + sparse: an index_put_ with accumulation; sparse +
//                  sparse: the entries of both, concatenated
//   mul            by a scalar (0-dim tensor): the values; sparse * sparse:
//                  the entries whose keys both have (a sorted search)
//   addmm / mm     sparse [n, m] @ dense [m, k]: each entry's row of the
//                  dense matrix, scaled, summed into its output row
//                  (index_add)
//   sparse_mask    a dense tensor's values at a sparse tensor's indices
// Compressed (CSR) tensors likewise, with the COO-CSR index conversions as
// searchsorted / repeat_interleave and their products through COO.

#include "vk.h"

#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <limits>

namespace vk {
namespace {

// `t` with row-major strides, every one canonical (PyTorch's compressed
// layouts check stride(-1) == 1, which contiguous() does not give a
// one-element or empty tensor).
at::Tensor packed(const at::Tensor& t) {
  int64_t expect = 1;
  bool canonical = true;
  for (int64_t d = t.dim() - 1; d >= 0; --d) {
    canonical = canonical && t.stride(d) == expect;
    expect *= std::max<int64_t>(t.size(d), 1);
  }
  return canonical ? t : at::empty(t.sizes(), t.options()).copy_(t);
}

at::Tensor make_sparse(const at::Tensor& like, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size,
                       const at::Tensor& indices, const at::Tensor& values, std::optional<bool> coalesced) {
  return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values,
                                                     like.options().dtype(values.scalar_type()).layout(at::kSparse),
                                                     coalesced);
}

// One int64 key per entry: its indices, row-major in the sparse dims.
at::Tensor linear_keys(const at::Tensor& indices, at::IntArrayRef size, int64_t sparse_dim) {
  at::Tensor key = at::zeros({indices.size(1)}, indices.options());
  for (int64_t d = 0; d < sparse_dim; ++d) {
    key = key * size[d] + indices.select(0, d);
  }
  return key;
}

// The indices [sparse_dim, n] of row-major keys.
at::Tensor unlinear_keys(at::Tensor key, at::IntArrayRef size, int64_t sparse_dim) {
  std::vector<at::Tensor> rows(sparse_dim);
  for (int64_t d = sparse_dim - 1; d >= 0; --d) {
    rows[d] = at::remainder(key, size[d]);
    key = at::div(key, size[d], "floor");
  }
  return sparse_dim == 0 ? at::empty({0, key.numel()}, key.options()) : at::stack(rows);
}

at::Tensor coalesce(const at::Tensor& self) {
  TORCH_CHECK(self.layout() == at::kSparse, "coalesce: expected a sparse COO tensor");
  int64_t sd = self.sparse_dim(), dd = self.dense_dim(), nnz = self._nnz();
  at::Tensor indices = self._indices(), values = self._values();
  if (nnz < 2) {
    return make_sparse(self, sd, dd, self.sizes(), indices.clone(), values.clone(), true);
  }
  if (sd == 0) {
    return make_sparse(self, sd, dd, self.sizes(), at::empty({0, 1}, indices.options()), values.sum(0, true), true);
  }
  at::Tensor key = linear_keys(indices, self.sizes(), sd);
  // (already in order without repeats, as most tensors built by these ops:
  // one comparison, no sort, no sums)
  if (at::gt(key.narrow(0, 1, nnz - 1), key.narrow(0, 0, nnz - 1)).all().item<bool>()) {
    return make_sparse(self, sd, dd, self.sizes(), indices.clone(), values.clone(), true);
  }
  // one stable sort of the keys; each entry's group from the runs; the values
  // summed by group in index order (one launch), as the CPU's coalesce
  at::Tensor order = sort_rows(key.reshape({1, nnz}).contiguous(), false).reshape({nnz});
  at::Tensor ol = order.to(at::kLong);
  at::Tensor sorted = key.index_select(0, ol);
  at::Tensor head = at::cat({at::ones({1}, key.options().dtype(at::kBool)),
                             at::ne(sorted.narrow(0, 1, nnz - 1), sorted.narrow(0, 0, nnz - 1))});
  at::Tensor group_sorted = at::cumsum(head, 0, at::kLong) - 1;
  int64_t u = group_sorted[nnz - 1].item<int64_t>() + 1;
  at::Tensor group = at::empty({nnz}, key.options()).index_copy_(0, ol, group_sorted);
  at::Tensor unique_keys = at::zeros({u}, key.options()).index_copy_(0, group_sorted, sorted);  // (a run's keys agree)
  std::vector<int64_t> vshape(values.sizes().begin(), values.sizes().end());
  vshape[0] = u;
  at::Tensor summed = sum_rows_by_sorted_group(values.reshape({nnz, -1}), group, order, u).view(vshape);
  return make_sparse(self, sd, dd, self.sizes(), unlinear_keys(unique_keys, self.sizes(), sd), summed, true);
}

// ---------------------------------------------------------------- add, mul

at::Tensor& add_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
  at::ScalarType t = at::result_type(self, other);
  if (!self.is_sparse() && other.is_sparse()) {  // dense + sparse: dense
    TORCH_CHECK(self.sizes() == other.sizes(), "add: expected 'self' and 'other' to have same size, but self has size ",
                self.sizes(), " while other has size ", other.sizes(),
                " (FYI: dense-sparse addition does not currently support broadcasting)");
    TORCH_CHECK(!out.is_sparse(), "add: expected a dense result for dense + sparse");
    if (!out.is_same(self)) {
      at::native::resize_output(out, self.sizes());
      out.copy_(self);
    }
    if (other._nnz() == 0) return out;
    at::Tensor values = other._values().to(out.scalar_type());
    if (!(alpha.isIntegral(true) && alpha.toLong() == 1)) values = values * alpha;
    if (other.sparse_dim() == 0) return out.add_(values.sum(0));  // (every entry is the whole tensor)
    at::Tensor idx = other._indices();
    c10::List<std::optional<at::Tensor>> rows;
    for (int64_t d = 0; d < other.sparse_dim(); ++d) rows.push_back(idx.select(0, d));
    out.index_put_(rows, values, /*accumulate=*/true);
    return out;
  }
  TORCH_CHECK(self.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(other.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(self.sizes() == other.sizes(), "add: expected sizes of 'self' and 'other' to match, but ",
              self.sizes(), " != ", other.sizes());
  if (other._nnz() == 0 || self._nnz() == 0) {  // one side empty: the other (any sparse dims), as PyTorch's
    at::Tensor r = other._nnz() == 0 ? self : other;
    at::Tensor v = r._values().to(t);
    if (other._nnz() != 0 && !(alpha.isIntegral(true) && alpha.toLong() == 1)) v = v * alpha;
    r = make_sparse(r, r.sparse_dim(), r.dense_dim(), r.sizes(), r._indices().clone(), v, r.is_coalesced());
    out.copy_(r);
    out._coalesced_(r.is_coalesced());
    return out;
  }
  TORCH_CHECK(self.sparse_dim() == other.sparse_dim(), "add: expected 'self' and 'other' to have same sparse dimensions");
  at::Tensor ov = other._values().to(t);
  if (!(alpha.isIntegral(true) && alpha.toLong() == 1)) ov = ov * alpha;
  if (self.is_coalesced() && other.is_coalesced() && self.dense_dim() == other.dense_dim()) {
    // Two coalesced operands: merged in key order without a sort (each
    // entry's place: its index plus the other side's keys before it), a
    // key in both summed (self's value first, as the CPU's merge).
    int64_t sd = self.sparse_dim(), na = self._nnz(), nb = other._nnz();
    at::Tensor ka = linear_keys(self._indices(), self.sizes(), sd), kb = linear_keys(other._indices(), self.sizes(), sd);
    at::TensorOptions lo = ka.options();
    at::Tensor pa = at::arange(na, lo) + at::searchsorted(kb, ka);
    at::Tensor pb = at::arange(nb, lo) + at::searchsorted(ka, kb, false, true);
    int64_t n = na + nb;
    at::Tensor keys = at::empty({n}, lo).index_copy_(0, pa, ka).index_copy_(0, pb, kb);
    std::vector<int64_t> vshape(ov.sizes().begin(), ov.sizes().end());
    vshape[0] = n;
    at::Tensor vals = at::empty(vshape, ov.options()).index_copy_(0, pa, self._values().to(t)).index_copy_(0, pb, ov);
    at::Tensor head = at::cat({at::ones({1}, lo.dtype(at::kBool)), at::ne(keys.narrow(0, 1, n - 1), keys.narrow(0, 0, n - 1))});
    at::Tensor group = at::cumsum(head, 0, at::kLong) - 1;
    int64_t u = group[n - 1].item<int64_t>() + 1;
    if (u < n) {
      vshape[0] = u;
      vals = sum_rows_by_sorted_group(vals.reshape({n, -1}), group, at::arange(n, lo), u).view(vshape);
      keys = at::zeros({u}, lo).index_copy_(0, group, keys);
    }
    at::Tensor r = make_sparse(self, sd, self.dense_dim(), self.sizes(), unlinear_keys(keys, self.sizes(), sd), vals, true);
    out.copy_(r);
    out._coalesced_(true);
    return out;
  }
  at::Tensor indices = at::cat({self._indices(), other._indices()}, 1);
  at::Tensor values = at::cat({self._values().to(t), ov}, 0);
  at::Tensor r = make_sparse(self, self.sparse_dim(), self.dense_dim(), self.sizes(), indices, values, false);
  // coalesced operands give a coalesced sum, as PyTorch's merge (nnz does not grow with repeated adds)
  if (self.is_coalesced() && other.is_coalesced()) r = r.coalesce();
  out.copy_(r);
  out._coalesced_(r.is_coalesced());
  return out;
}

// sparse * sparse: the entries at the keys both have (after coalescing);
// the values broadcast in the dense dims.
at::Tensor mul_sparse_sparse(const at::Tensor& a0, const at::Tensor& b0) {
  bool same_sparse = a0.sparse_dim() == b0.sparse_dim() && a0.dim() == b0.dim() &&
                     a0.sizes().slice(0, a0.sparse_dim()) == b0.sizes().slice(0, b0.sparse_dim());
  TORCH_CHECK(same_sparse, "mul(sparse, sparse): expected the same sparse dims and sizes, but got ", a0.sizes(),
              " and ", b0.sizes());
  std::vector<int64_t> res_shape = at::infer_size(a0.sizes(), b0.sizes());
  at::ScalarType t = at::result_type(a0, b0);
  at::Tensor a = a0.coalesce(), b = b0.coalesce();
  int64_t sd = a.sparse_dim();
  at::Tensor ka = linear_keys(a._indices(), a.sizes(), sd), kb = linear_keys(b._indices(), b.sizes(), sd);
  std::vector<int64_t> vshape = {0};
  vshape.insert(vshape.end(), res_shape.begin() + sd, res_shape.end());
  if (ka.numel() == 0 || kb.numel() == 0) {
    return make_sparse(a, sd, a.dense_dim(), res_shape, at::empty({sd, 0}, ka.options()),
                       at::empty(vshape, a._values().options().dtype(t)), true);
  }
  at::Tensor pos = at::searchsorted(kb, ka).clamp_max(kb.numel() - 1);
  at::Tensor hit = at::eq(kb.index_select(0, pos), ka);
  at::Tensor ia = at::nonzero(hit).reshape({-1});
  at::Tensor ib = pos.index_select(0, ia);
  at::Tensor values = a._values().index_select(0, ia).to(t) * b._values().index_select(0, ib).to(t);
  return make_sparse(a, sd, a.dense_dim(), res_shape, a._indices().index_select(1, ia), values, true);
}

// sparse * dense, broadcasting, as PyTorch's intersection of a sparse COO
// tensor with a dense one: the dense values at the sparse entries (a dense
// operand with more dims repeats the entries along its leading dims).
at::Tensor mul_sparse_dense(const at::Tensor& s0, const at::Tensor& d, at::ScalarType t, bool in_place) {
  std::vector<int64_t> res_shape = at::infer_size(d.sizes(), s0.sizes());
  int64_t dd = s0.dense_dim();
  if (!s0._nnz() || !s0.numel() || !d.numel()) {
    int64_t sd = static_cast<int64_t>(res_shape.size()) - dd;
    std::vector<int64_t> vshape(s0._values().sizes().begin(), s0._values().sizes().end());
    vshape[0] = 0;
    return make_sparse(s0, sd, dd, res_shape, at::empty({sd, 0}, s0._indices().options()),
                       at::empty(vshape, s0._values().options().dtype(t)), true);
  }
  int64_t d_dim = d.dim(), s_dim = s0.dim();
  at::Tensor s = d_dim > s_dim ? s0.coalesce() : s0;
  int64_t sd = s.sparse_dim();
  at::Tensor si = s._indices(), sv = s._values();
  auto values_of = [&](const at::Tensor& df) {
    at::Tensor v = at::mul(df, sv);
    return in_place ? v : v.to(t);
  };
  if (d_dim <= dd) {
    return make_sparse(s, sd, dd, res_shape, si.clone(), values_of(d), s.is_coalesced());
  }
  int64_t inter = std::min(sd, d_dim - dd);
  int64_t d_start = std::max<int64_t>(0, d_dim - s_dim), s_start = std::max<int64_t>(0, s_dim - d_dim);
  std::vector<at::indexing::TensorIndex> idx;
  if (d_start) idx.emplace_back(at::indexing::Ellipsis);
  for (int64_t i = 0; i < inter; ++i) idx.emplace_back(si[s_start + i]);
  for (int64_t i = d_start + inter; i < d_dim; ++i) idx.emplace_back(at::indexing::Slice());
  std::vector<int64_t> d_shape(res_shape.end() - d_dim, res_shape.end());
  at::Tensor df = d.expand(d_shape).index(idx);
  if (s_dim >= d_dim) {
    return make_sparse(s, sd, dd, res_shape, si.clone(), values_of(df), s.is_coalesced());
  }
  // d's leading (batch) dims become new sparse dims: entry k * nnz + j is
  // batch k (row-major) of s's entry j
  int64_t nnz = s._nnz(), batches = 1;
  for (int64_t i = 0; i < d_start; ++i) batches *= d.size(i);
  std::vector<int64_t> vshape(sv.sizes().begin(), sv.sizes().end());
  vshape[0] = batches * nnz;
  at::Tensor values = values_of(df).reshape(vshape);
  at::Tensor e = at::arange(batches * nnz, si.options());
  at::Tensor k = at::floor_divide(e, nnz), j = at::remainder(e, nnz);
  std::vector<at::Tensor> batch(d_start);  // (d_start >= 1 here)
  for (int64_t i = d_start - 1; i >= 0; --i) {
    batch[i] = at::remainder(k, d.size(i));
    k = at::floor_divide(k, d.size(i));
  }
  at::Tensor indices = at::cat({at::stack(batch), si.index_select(1, j)}, 0);
  return make_sparse(s, d_start + sd, dd, res_shape, indices, values, true);
}

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  at::ScalarType t = at::result_type(self, other);
  at::Tensor r;
  if (self.is_sparse() && other.is_sparse()) {
    if (self.dim() == 0 || other.dim() == 0) {
      // a 0-dim operand as a dense scalar, as PyTorch picks it: `other` if
      // 0-dim, else `self`; the other one instead if that is a coalesced
      // 0-dim tensor and this one is not
      bool z_is_other = other.dim() == 0;
      if (!(z_is_other ? other : self).is_coalesced() && (z_is_other ? self : other).dim() == 0 &&
          (z_is_other ? self : other).is_coalesced()) {
        z_is_other = !z_is_other;
      }
      const at::Tensor& z = z_is_other ? other : self;
      const at::Tensor& s = z_is_other ? self : other;
      at::Tensor c = z.coalesce();
      at::Tensor scalar = c._nnz() == 0 ? at::zeros({}, c._values().options()) : c._values().sum(0);
      r = mul_sparse_dense(s, scalar, t, false);
    } else {
      r = mul_sparse_sparse(self, other);
    }
  } else {
    const at::Tensor& s = self.is_sparse() ? self : other;
    const at::Tensor& d = self.is_sparse() ? other : self;
    r = mul_sparse_dense(s, d, out.is_same(s) ? out.scalar_type() : t, out.is_same(s));
  }
  if (out.is_sparse()) {
    out.copy_(r);
    out._coalesced_(r.is_coalesced());
  } else {
    at::native::resize_output(out, r.sizes());
    out.copy_(r.to_dense());
  }
  return out;
}

// ---------------------------------------------------------------- matrix products

// beta self + alpha (sparse [n, m] @ dense [m, k]) into out (dense [n, k]).
at::Tensor& addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  TORCH_CHECK(mat1.is_sparse() && !mat2.is_sparse(), "addmm: expected a sparse mat1 and a dense mat2");
  TORCH_CHECK(mat1.sparse_dim() == 2 && mat1.dense_dim() == 0, "addmm: matrices expected, got ", mat1.sparse_dim(),
              "D sparse tensor");
  TORCH_CHECK(mat2.dim() == 2, "addmm: matrices expected, got ", mat2.dim(), "D tensor");
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (", mat1.size(0), "x",
              mat1.size(1), " and ", mat2.size(0), "x", mat2.size(1), ")");
  int64_t n = mat1.size(0), k = mat2.size(1);
  at::ScalarType t = at::result_type(mat1, mat2);
  at::Tensor product = at::zeros({n, k}, mat2.options().dtype(t));
  int64_t nnz = mat1._nnz();
  if (nnz > 0) {
    // each entry's scaled row of mat2 added into its output row, in entry
    // order; in chunks of entries, so the gathered rows stay within 2^22
    // elements (nnz * k of them at once: gigabytes for a 10^5-entry matrix
    // times 1000 columns)
    // (each chunk: one sort of its output rows, then whole rows folded in
    // entry order by sum_rows_by_sorted_group; complex rows as real pairs)
    at::Tensor idx = mat1._indices(), vals = mat1._values().to(t), m2 = mat2.to(t);
    bool cplx = at::isComplexType(t);
    int64_t width = cplx ? 2 * k : k;
    int64_t chunk = std::max<int64_t>(1, (int64_t(1) << 22) / std::max<int64_t>(width, 1));
    for (int64_t start = 0; start < nnz; start += chunk) {
      int64_t len = std::min(chunk, nnz - start);
      at::Tensor out_rows = idx.select(0, 0).narrow(0, start, len).contiguous();
      at::Tensor rows = m2.index_select(0, idx.select(0, 1).narrow(0, start, len)) * vals.narrow(0, start, len).unsqueeze(1);
      at::Tensor flat = cplx ? at::view_as_real(rows).reshape({len, width}) : rows;
      at::Tensor order = sort_rows(out_rows.reshape({1, len}), false).reshape({len});
      at::Tensor summed = sum_rows_by_sorted_group(flat, out_rows, order, n);  // [n, width]
      product.add_(cplx ? at::view_as_complex(summed.view({n, k, 2})) : summed);
    }
  }
  at::Tensor base = self.expand({n, k});
  at::Tensor r = beta.toComplexDouble() == 0.0 ? product * alpha : at::add(base * beta, product, alpha);
  if (!out.is_same(self)) at::native::resize_output(out, {n, k});
  out.copy_(r);
  return out;
}

at::Tensor addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                 const at::Scalar& alpha) {
  at::Tensor out = at::empty({0}, mat2.options().dtype(at::result_type(mat1, mat2)));
  return addmm_out(self, mat1, mat2, beta, alpha, out);
}

at::Tensor& addmm_(at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                   const at::Scalar& alpha) {
  return addmm_out(self, mat1, mat2, beta, alpha, self);
}

// sparse [n, m] @ sparse [m, p], no dense dims, from the entries: each
// entry (i, k) of `a` meets row k of `b` (found by a sorted search), the
// products coalesced. Memory grows with the products' count, not n * p.
// Products that cancel stay as entries (zeros), as the CPU's.
at::Tensor spgemm(const at::Tensor& a0, const at::Tensor& b0) {
  at::Tensor a = a0.coalesce(), b = b0.coalesce();
  int64_t n = a.size(0), m = a.size(1), p = b.size(1);
  at::Tensor ai = a._indices(), bi = b._indices();
  at::TensorOptions lopts = ai.options();
  int64_t total = 0;
  at::Tensor start, counts;
  if (a._nnz() > 0 && b._nnz() > 0) {
    at::Tensor bptr = at::searchsorted(bi.select(0, 0).contiguous(), at::arange(m + 1, lopts));  // (b's rows sorted)
    at::Tensor ak = ai.select(0, 1);
    start = bptr.index_select(0, ak);
    counts = bptr.index_select(0, ak + 1) - start;
    total = counts.sum().item<int64_t>();
  }
  if (total == 0) {
    return make_sparse(a, 2, 0, {n, p}, at::empty({2, 0}, lopts), at::empty({0}, a._values().options()), true);
  }
  at::Tensor rep = at::repeat_interleave(at::arange(a._nnz(), lopts), counts, 0, total);
  at::Tensor first = at::cumsum(counts, 0) - counts;  // each A entry's first product
  at::Tensor pos = at::arange(total, lopts) - first.index_select(0, rep) + start.index_select(0, rep);
  at::Tensor rows = ai.select(0, 0).index_select(0, rep), cols = bi.select(0, 1).index_select(0, pos);
  at::Tensor vals = a._values().index_select(0, rep) * b._values().index_select(0, pos);
  return make_sparse(a, 2, 0, {n, p}, at::stack({rows, cols}), vals, false).coalesce();
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "mm: expected 2-D tensors");
  if (self.is_sparse() && mat2.is_sparse()) {  // sparse @ sparse: sparse
    TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "mat1 dtype ", self.scalar_type(),
                " does not match mat2 dtype ", mat2.scalar_type());
    TORCH_CHECK(self.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (", self.size(0), "x",
                self.size(1), " and ", mat2.size(0), "x", mat2.size(1), ")");
    if (self.dense_dim() == 0 && mat2.dense_dim() == 0) return spgemm(self, mat2);
    return vk::mm(self, mat2.to_dense()).to_sparse();
  }
  if (!self.is_sparse()) {  // dense @ sparse = (sparse^T @ dense^T)^T
    return vk::mm(mat2.t(), self.t()).t();
  }
  at::Tensor zero = at::zeros({}, mat2.options().dtype(at::result_type(self, mat2)));
  return vk::addmm(zero, self, mat2, 0, 1);
}

// ---------------------------------------------------------------- sparse_mask

at::Tensor sparse_mask(const at::Tensor& self, const at::Tensor& mask) {
  TORCH_CHECK(mask.is_sparse(), "sparse_mask expects mask to be sparse");
  TORCH_CHECK(self.sizes() == mask.sizes(), "sparse_mask(): operands have incompatible sizes; self has size ",
              self.sizes(), " but mask has size ", mask.sizes());
  const at::Tensor& m = mask;  // (its entries as they are: the result is coalesced if the mask is)
  at::Tensor src = self.is_sparse() ? self.to_dense() : self;
  c10::List<std::optional<at::Tensor>> rows;
  for (int64_t d = 0; d < m.sparse_dim(); ++d) rows.push_back(m._indices().select(0, d));
  at::Tensor values = m._nnz() == 0 || m.sparse_dim() == 0 ? at::empty({0}, src.options()) : at::index(src, rows);
  if (m.sparse_dim() == 0) values = src.unsqueeze(0).expand(m._values().sizes()).clone();
  if (m._nnz() == 0) {
    std::vector<int64_t> vshape(m._values().sizes().begin(), m._values().sizes().end());
    values = at::empty(vshape, src.options());
  }
  return make_sparse(m, m.sparse_dim(), m.dense_dim(), m.sizes(), m._indices().clone(), values, m.is_coalesced());
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  using namespace at::native;
  // (device-generic parts of PyTorch's sparse implementation)
  m.impl("_sparse_coo_tensor_with_dims_and_tensors", TORCH_FN(new_with_dims_and_tensor_sparse_symint));
  m.impl("empty.memory_format", TORCH_FN(empty_sparse_symint));
  m.impl("empty_like", TORCH_FN(empty_like_sparse_coo));
  m.impl("_indices", TORCH_FN(_indices_sparse));
  m.impl("_values", TORCH_FN(_values_sparse));
  m.impl("indices", TORCH_FN(indices_sparse));
  m.impl("values", TORCH_FN(values_sparse));
  m.impl("_nnz", TORCH_FN(_nnz_sparse));
  m.impl("sparse_dim", TORCH_FN(sparse_dim_sparse));
  m.impl("dense_dim", TORCH_FN(dense_dim_sparse));
  m.impl("is_coalesced", TORCH_FN(is_coalesced_sparse));
  m.impl("_coalesced_", TORCH_FN(_coalesced_sparse_));
  m.impl("sparse_resize_", TORCH_FN(sparse_resize_));
  m.impl("sparse_resize_and_clear_", TORCH_FN(sparse_resize_and_clear_));
  m.impl("resize_as_sparse_", TORCH_FN(resize_as_sparse_));
  m.impl("copy_", TORCH_FN(copy_sparse_wrapper_));
  m.impl("copy_sparse_to_sparse_", TORCH_FN(copy_sparse_));
  m.impl("clone", TORCH_FN(clone_sparse));
  m.impl("zero_", TORCH_FN(zero_sparse_));
  m.impl("_to_dense", TORCH_FN(sparse_to_dense));
  m.impl("_to_sparse.sparse_dim", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, int64_t)>(sparse_coo_to_sparse)));
  m.impl("sum", TORCH_FN(sum_coo));
  m.impl("cat", TORCH_FN(cat_sparse));
  m.impl("permute", TORCH_FN(permute_sparse_coo));
  m.impl("unsqueeze", TORCH_FN(unsqueeze_sparse));
  m.impl("narrow_copy", TORCH_FN(narrow_copy_sparse));
  m.impl("mv", TORCH_FN(mv_sparse));
  m.impl("norm.ScalarOpt_dim_dtype", TORCH_FN(sparse_dtype_norm));
  m.impl("norm.ScalarOpt_dim", TORCH_FN(sparse_norm));
  m.impl("native_norm", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, const at::Scalar&)>(norm_sparse)));
  m.impl("native_norm.ScalarOpt_dim_dtype", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, const std::optional<at::Scalar>&, at::IntArrayRef, bool, std::optional<at::ScalarType>)>(norm_sparse)));
  m.impl("any", TORCH_FN(any_sparse));
  m.impl("add.Tensor", TORCH_FN(add_sparse));
  m.impl("add_.Tensor", TORCH_FN(add_sparse_));
  m.impl("sub.Tensor", TORCH_FN(sub_sparse));
  m.impl("sub_.Tensor", TORCH_FN(sub_sparse_));
  m.impl("sub.out", TORCH_FN(sub_out_sparse));
  m.impl("mul.Tensor", TORCH_FN(mul_sparse));
  m.impl("mul_.Tensor", TORCH_FN(mul_sparse_));
  m.impl("div.Tensor", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&)>(div_sparse)));
  m.impl("div_.Tensor", TORCH_FN(static_cast<at::Tensor& (*)(at::Tensor&, const at::Tensor&)>(div_sparse_)));
  m.impl("div.out", TORCH_FN(static_cast<at::Tensor& (*)(const at::Tensor&, const at::Tensor&, at::Tensor&)>(
                        div_out_sparse_zerodim)));
  m.impl("div.Tensor_mode",
         TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&, std::optional<c10::string_view>)>(
             div_sparse)));
  m.impl("div_.Tensor_mode",
         TORCH_FN(static_cast<at::Tensor& (*)(at::Tensor&, const at::Tensor&, std::optional<c10::string_view>)>(
             div_sparse_)));
  m.impl("div.out_mode", TORCH_FN(static_cast<at::Tensor& (*)(const at::Tensor&, const at::Tensor&,
                                                              std::optional<c10::string_view>, at::Tensor&)>(
                             div_out_sparse_zerodim)));
  m.impl("floor_divide", TORCH_FN(floor_divide_sparse));
  m.impl("floor_divide_.Tensor", TORCH_FN(floor_divide_sparse_));
  m.impl("floor_divide.out", TORCH_FN(floor_divide_out_sparse_zerodim));
  m.impl("pow.Tensor_Scalar", TORCH_FN(pow_sparse_scalar));
  m.impl("pow.Tensor_Scalar_out", TORCH_FN(pow_out_sparse_scalar));
  m.impl("nan_to_num", TORCH_FN(nan_to_num_sparse));
  m.impl("nan_to_num_", TORCH_FN(nan_to_num_sparse_));
  m.impl("nan_to_num.out", TORCH_FN(nan_to_num_sparse_out));
  m.impl("threshold_backward", TORCH_FN(threshold_backward_sparse));
  m.impl("threshold_backward.grad_input", TORCH_FN(threshold_backward_sparse_out));
#define VK_SPARSE_UNARY(op)                             \
  m.impl(#op, TORCH_FN(op##_sparse));                   \
  m.impl(#op "_", TORCH_FN(op##_sparse_));              \
  m.impl(#op ".out", TORCH_FN(op##_sparse_out));
  VK_SPARSE_UNARY(abs)
  VK_SPARSE_UNARY(asin)
  VK_SPARSE_UNARY(asinh)
  VK_SPARSE_UNARY(atan)
  VK_SPARSE_UNARY(atanh)
  VK_SPARSE_UNARY(ceil)
  VK_SPARSE_UNARY(deg2rad)
  VK_SPARSE_UNARY(erf)
  VK_SPARSE_UNARY(erfinv)
  VK_SPARSE_UNARY(expm1)
  VK_SPARSE_UNARY(floor)
  VK_SPARSE_UNARY(frac)
  VK_SPARSE_UNARY(log1p)
  VK_SPARSE_UNARY(rad2deg)
  VK_SPARSE_UNARY(round)
  VK_SPARSE_UNARY(sgn)
  VK_SPARSE_UNARY(sign)
  VK_SPARSE_UNARY(sin)
  VK_SPARSE_UNARY(sinh)
  VK_SPARSE_UNARY(sqrt)
  VK_SPARSE_UNARY(tan)
  VK_SPARSE_UNARY(tanh)
  VK_SPARSE_UNARY(trunc)
#undef VK_SPARSE_UNARY
  m.impl("neg", TORCH_FN(neg_sparse));
  m.impl("neg_", TORCH_FN(neg_sparse_));
  m.impl("neg.out", TORCH_FN(neg_out_sparse));
  m.impl("relu", TORCH_FN(relu_sparse));
  m.impl("relu_", TORCH_FN(relu_sparse_));
  m.impl("isnan", TORCH_FN(isnan_sparse));
  m.impl("isinf", TORCH_FN(isinf_sparse));
  m.impl("isposinf", TORCH_FN(isposinf_sparse));
  m.impl("isneginf", TORCH_FN(isneginf_sparse));
  m.impl("signbit", TORCH_FN(signbit_sparse));
  // (the per-device kernels, in tensor ops)
  m.impl("_coalesce", TORCH_FN(vk::coalesce));
  m.impl("add.out", TORCH_FN(vk::add_out));
  m.impl("mul.out", TORCH_FN(vk::mul_out));
  m.impl("addmm", TORCH_FN(vk::addmm));
  m.impl("addmm.out", TORCH_FN(vk::addmm_out));
  m.impl("addmm_", TORCH_FN(vk::addmm_));
  m.impl("mm", TORCH_FN(vk::mm));
  m.impl("sparse_mask", TORCH_FN(vk::sparse_mask));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace at::native;
  // dense -> sparse (PyTorch's, in tensor ops)
  m.impl("_to_sparse.sparse_dim", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, int64_t)>(dense_to_sparse)));
  m.impl("_to_sparse", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, std::optional<at::Layout>, at::OptionalIntArrayRef, std::optional<int64_t>)>(dense_to_sparse)));
}

// ---------------------------------------------------------------- compressed (CSR) tensors

namespace vk {
namespace {

// crow [size + 1] of sorted row indices: the number of rows before each row.
at::Tensor convert_coo_to_csr(const at::Tensor& self, int64_t size, bool out_int32) {
  TORCH_CHECK(self.dim() <= 1, "Input is supposed to be a vector, but got ", self.dim(), " dimensional tensor.");
  TORCH_CHECK(size >= 0, "size must be non-negative but got ", size);
  at::Tensor rows = self.to(at::kLong).reshape({-1});
  at::Tensor bounds = at::arange(size + 1, rows.options());
  return at::searchsorted(rows, bounds).to(out_int32 ? at::kInt : at::kLong);
}

// [2, nnz] (row, col) indices of a CSR matrix's (crow, col): each row index
// repeated by its row's count; transposed: (col, row).
at::Tensor convert_csr_to_coo(const at::Tensor& crow, const at::Tensor& col, bool out_int32, bool transpose) {
  TORCH_CHECK(crow.dim() == col.dim(), "crow_indices and col_indices are supposed to have the same dimensionality, but got ",
              crow.dim(), " and ", col.dim(), " dimensional tensors, respectively.");
  at::ScalarType t = out_int32 ? at::kInt : at::kLong;
  if (crow.dim() > 1 && is_vk(crow) && is_vk(col)) {
    // batched: [batch dims..., row, col] per entry, the batches one after another
    int64_t nb = crow.dim() - 1, rows1 = crow.size(-1), nnz = col.size(-1);
    int64_t batches = 1;
    for (int64_t d = 0; d < nb; ++d) batches *= crow.size(d);
    at::Tensor c = crow.to(at::kLong).reshape({batches, rows1});
    at::Tensor entry = at::arange(nnz, c.options()).expand({batches, nnz}).contiguous();
    at::Tensor rows = at::searchsorted(c, entry, false, true) - 1;
    // (entries past a batch's last row start: row 0, as the CPU leaves them)
    rows = at::where(at::lt(entry, c.slice(1, rows1 - 1, rows1)), rows, at::zeros_like(rows));
    std::vector<at::Tensor> parts;
    at::Tensor b = at::arange(batches, c.options()).unsqueeze(1).expand({batches, nnz});
    int64_t inner = batches;
    for (int64_t d = 0; d < nb; ++d) {
      inner /= crow.size(d);
      parts.push_back(at::remainder(at::floor_divide(b, inner), crow.size(d)).reshape({-1}));
    }
    at::Tensor cols = col.to(at::kLong).reshape({-1});
    if (transpose) {
      parts.push_back(cols);
      parts.push_back(rows.reshape({-1}));
    } else {
      parts.push_back(rows.reshape({-1}));
      parts.push_back(cols);
    }
    return at::stack(parts).to(t);
  }
  if (crow.dim() != 1) {
    auto stack = on_cpu("aten::_convert_indices_from_csr_to_coo", "", crow, col, out_int32, transpose);
    return stack[0].toTensor();
  }
  at::Tensor c = crow.to(at::kLong);
  // each entry's row: the last row starting at or before it
  at::Tensor rows = at::searchsorted(c, at::arange(col.numel(), c.options()), false, true) - 1;
  at::Tensor cols = col.to(at::kLong);
  return (transpose ? at::stack({cols, rows}) : at::stack({rows, cols})).to(t);
}

// A CSR matrix as a COO one (same values).
at::Tensor csr_as_coo(const at::Tensor& csr) {
  at::Tensor idx = convert_csr_to_coo(csr.crow_indices(), csr.col_indices(), false, false);
  return make_sparse(csr, 2, 0, csr.sizes(), idx, csr.values(), true);
}

// A 2-D COO or CSR matrix (or a dense one) as CSC: entries ordered by
// (column, row), compressed by column.
at::Tensor to_csc(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  TORCH_CHECK(self.dim() == 2 && (!dense_dim.has_value() || *dense_dim == 0),
              "torch_vulkan: to_sparse_csc of 2-D matrices without dense dims only");
  at::Tensor coo = self.layout() == at::kSparseCsr ? csr_as_coo(self)
                   : self.layout() == at::kSparse  ? self
                                                   : self.to_sparse();
  at::Tensor t = make_sparse(coo, 2, 0, {self.size(1), self.size(0)}, coo._indices().flip(0), coo._values(), false)
                     .coalesce();
  at::Tensor ccol = convert_coo_to_csr(t._indices().select(0, 0), self.size(1), false);
  return at::_sparse_compressed_tensor_unsafe(packed(ccol), packed(t._indices().select(0, 1)), packed(t._values()), self.sizes(),
                                              t._values().options().layout(at::kSparseCsc));
}

at::Tensor to_csc_op(const at::Tensor& self, std::optional<int64_t> dense_dim) { return to_csc(self, dense_dim); }

at::Tensor to_layout(const at::Tensor& self, std::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize,
                     std::optional<int64_t> dense_dim);

// The specified elements of a compressed tensor (explicit zeros included),
// as COO over its batch and matrix dims: in storage order (blocks, then
// row-major within a block), coalesced for CSR/BSR as PyTorch's.
at::Tensor compressed_entries(const at::Tensor& t) {
  at::Layout l = t.layout();
  bool by_col = l == at::kSparseCsc || l == at::kSparseBsc;
  bool blocked = l == at::kSparseBsr || l == at::kSparseBsc;
  int64_t dd = t.dense_dim(), nb = t.dim() - 2 - dd;
  at::Tensor ci = by_col ? t.ccol_indices() : t.crow_indices();
  at::Tensor pi = by_col ? t.row_indices() : t.col_indices();
  at::Tensor vals = t.values();  // [*batch, nnz, (R, C,) *dense]
  int64_t B = 1;
  for (int64_t d = 0; d < nb; ++d) B *= t.size(d);
  int64_t outer = ci.size(-1) - 1, nnz = pi.size(-1);
  int64_t R = blocked ? vals.size(nb + 1) : 1, C = blocked ? vals.size(nb + 2) : 1;
  int64_t E = B * nnz, N = E * R * C;
  at::TensorOptions lopts = pi.options().dtype(at::kLong);
  std::vector<int64_t> dshape(t.sizes().end() - dd, t.sizes().end());
  std::vector<int64_t> vshape = {N};
  vshape.insert(vshape.end(), dshape.begin(), dshape.end());
  at::Tensor flat_vals = vals.reshape(vshape);
  at::Tensor indices;
  if (E == 0 || outer == 0) {
    indices = at::empty({nb + 2, 0}, lopts);
  } else {
    at::Tensor c2 = ci.reshape({B, outer + 1}).to(at::kLong);
    // each entry's batch * outer + o: a sorted search in the batches' starts,
    // offset to global positions
    at::Tensor starts = c2.narrow(1, 0, outer);
    if (B > 1) starts = starts + (at::arange(B, lopts) * nnz).view({B, 1});
    at::Tensor bo = at::searchsorted(starts.reshape({B * outer}), at::arange(E, lopts), false, true) - 1;
    at::Tensor o = B > 1 ? at::remainder(bo, outer) : bo;
    at::Tensor in = pi.reshape({E}).to(at::kLong);
    at::Tensor rb = by_col ? in : o, cb = by_col ? o : in;
    at::Tensor rows = rb, cols = cb;
    if (blocked) {
      rows = (rb.view({E, 1, 1}) * R + at::arange(R, lopts).view({1, R, 1})).expand({E, R, C}).reshape({N});
      cols = (cb.view({E, 1, 1}) * C + at::arange(C, lopts).view({1, 1, C})).expand({E, R, C}).reshape({N});
    }
    std::vector<at::Tensor> parts(nb + 2);
    if (nb > 0) {
      at::Tensor bb = at::floor_divide(bo, outer);
      if (blocked) bb = bb.view({E, 1, 1}).expand({E, R, C}).reshape({N});
      for (int64_t d = nb - 1; d >= 0; --d) {
        parts[d] = at::remainder(bb, t.size(d));
        bb = at::floor_divide(bb, t.size(d));
      }
    }
    parts[nb] = rows;
    parts[nb + 1] = cols;
    indices = at::stack(parts);
  }
  at::Tensor r = make_sparse(t, nb + 2, dd, t.sizes(), indices, flat_vals, false);
  return by_col ? r : r.coalesce();
}

// A compressed tensor as dense: its entries copied into a zeroed tensor
// (positions are unique in compressed layouts).
at::Tensor compressed_to_dense(const at::Tensor& self, std::optional<at::ScalarType> dtype,
                               std::optional<bool> masked_grad) {
  if (dtype.has_value() && *dtype != self.scalar_type()) {
    return at::native::sparse_compressed_to_dense(self, dtype, masked_grad);
  }
  at::Layout l = self.layout();
  if ((l == at::kSparseCsr || l == at::kSparseCsc) && self.dim() == 2 && self.dense_dim() == 0) {
    // (one matrix: each entry's outer index by a sorted search, then its
    // position in the dense matrix)
    bool by_col = l == at::kSparseCsc;
    at::Tensor c = (by_col ? self.ccol_indices() : self.crow_indices()).to(at::kLong);
    at::Tensor in = (by_col ? self.row_indices() : self.col_indices()).to(at::kLong);
    at::Tensor dense = at::zeros(self.sizes(), self.values().options());
    if (in.numel() == 0) return dense;
    at::Tensor o = at::searchsorted(c, at::arange(in.numel(), c.options()), false, true) - 1;
    int64_t N = self.size(1);
    dense.view({-1}).index_copy_(0, by_col ? in * N + o : o * N + in, self.values());
    return dense;
  }
  at::Tensor e = compressed_entries(self);
  int64_t sd = e.sparse_dim();
  at::Tensor dense = at::zeros(self.sizes(), e._values().options());
  if (e._nnz() == 0) return dense;
  int64_t inner = 1;
  for (int64_t d = sd; d < self.dim(); ++d) inner *= self.size(d);
  at::Tensor key = linear_keys(e._indices(), self.sizes(), sd);
  dense.view({-1, inner}).index_copy_(0, key, e._values().reshape({e._nnz(), inner}));
  return dense;
}

// A compressed matrix (CSR, CSC, BSR, BSC) as COO, explicit zeros kept.
at::Tensor compressed_as_coo(const at::Tensor& t) {
  if (t.layout() == at::kSparseCsr && t.dim() == 2 && t.dense_dim() == 0) return csr_as_coo(t);
  if (t.dim() - t.dense_dim() == 2) return compressed_entries(t);
  return t.to_dense().to_sparse(2);
}

bool compressed_layout(const at::Tensor& t) {
  at::Layout l = t.layout();
  return l == at::kSparseCsr || l == at::kSparseCsc || l == at::kSparseBsr || l == at::kSparseBsc;
}

at::Tensor dense_of(const at::Tensor& t) { return t.layout() == at::kStrided ? t : t.to_dense(); }

// a @ b for dense and 2-D compressed operands (as COO, explicit zeros
// kept): a dense result.
at::Tensor any_mm(const at::Tensor& a, const at::Tensor& b) {
  at::Tensor x = compressed_layout(a) ? compressed_as_coo(a) : a;
  at::Tensor y = compressed_layout(b) ? compressed_as_coo(b) : b;
  if (!x.is_sparse() && !y.is_sparse()) return at::mm(x, y);
  return dense_of(vk::mm(x, y));
}

// Replace compressed `out`'s contents with compressed `r` (of any nnz).
void assign_compressed(at::Tensor& out, const at::Tensor& r) {
  if (out.layout() == r.layout() && out._nnz() == r._nnz() && out.sizes() == r.sizes()) {
    out.copy_(r);
    return;
  }
  bool by_col = r.layout() == at::kSparseCsc || r.layout() == at::kSparseBsc;
  at::Tensor ci = by_col ? r.ccol_indices() : r.crow_indices(), pi = by_col ? r.row_indices() : r.col_indices();
  at::Tensor oci = by_col ? out.ccol_indices() : out.crow_indices();
  at::sparse_csr::get_sparse_csr_impl(out)->set_member_tensors(
      packed(ci.to(oci.scalar_type())), packed(pi.to(oci.scalar_type())), packed(r.values().to(out.scalar_type())),
      r.sizes());
}

// beta self + alpha (mat1 @ mat2) for any mix of dense and compressed
// operands (invalid shapes: the dense op's error).
at::Tensor addmm_any(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                     const at::Scalar& alpha) {
  bool shapes_ok = mat1.dim() == 2 && mat2.dim() == 2 && self.dim() <= 2 && mat1.size(1) == mat2.size(0) &&
                   (!compressed_layout(mat1) || mat1.dense_dim() == 0) &&
                   (!compressed_layout(mat2) || mat2.dense_dim() == 0);
  if (!shapes_ok) return at::addmm(dense_of(self), dense_of(mat1), dense_of(mat2), beta, alpha);
  at::Tensor r = any_mm(mat1, mat2);
  if (!(alpha.isIntegral(true) && alpha.toLong() == 1)) r = r * alpha;
  bool beta_zero = beta.isComplex() ? beta.toComplexDouble() == c10::complex<double>(0, 0) : beta.toDouble() == 0;
  if (!beta_zero) r = r + dense_of(self) * beta;  // (beta == 0 ignores self, NaN included)
  return r.expand({mat1.size(0), mat2.size(1)});
}

at::Tensor& csr_addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                          const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  at::Tensor r = addmm_any(self, mat1, mat2, beta, alpha);
  if (compressed_layout(out)) {
    std::optional<std::vector<int64_t>> bs;
    if (out.layout() == at::kSparseBsr || out.layout() == at::kSparseBsc) {
      bs = std::vector<int64_t>{out.values().size(-2), out.values().size(-1)};
    }
    assign_compressed(out, to_layout(r.contiguous(), out.layout(),
                                     bs.has_value() ? at::OptionalIntArrayRef(*bs) : std::nullopt, std::nullopt));
    return out;
  }
  at::native::resize_output(out, r.sizes());
  out.copy_(r);
  return out;
}

at::Tensor csr_addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                     const at::Scalar& alpha) {
  at::Tensor r = addmm_any(self, mat1, mat2, beta, alpha).contiguous();
  if (compressed_layout(mat1) && compressed_layout(mat2)) {  // all sparse: in mat1's layout, as PyTorch's
    return to_layout(r, mat1.layout(), std::nullopt, std::nullopt);
  }
  return r;
}

at::Tensor csr_mm(const at::Tensor& self, const at::Tensor& mat2) {
  bool shapes_ok = self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0);
  if (!shapes_ok) return at::mm(dense_of(self), dense_of(mat2));  // (the dense op's error)
  if (compressed_layout(self) && compressed_layout(mat2) && self.dense_dim() == 0 && mat2.dense_dim() == 0) {
    // CSR, as PyTorch's: the sparse product of the entries
    return to_layout(spgemm(compressed_as_coo(self), compressed_as_coo(mat2)), at::kSparseCsr, std::nullopt,
                     std::nullopt);
  }
  at::Tensor r = any_mm(self, mat2);
  if (compressed_layout(self) && compressed_layout(mat2)) {
    return to_layout(r.contiguous(), at::kSparseCsr, std::nullopt, std::nullopt);
  }
  return r;
}

// beta self + alpha (mat @ vec), mat compressed.
at::Tensor& csr_addmv_out(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta,
                          const at::Scalar& alpha, at::Tensor& out) {
  TORCH_CHECK(mat.dim() == 2 && vec.dim() == 1, "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(),
              ", ", vec.dim());
  TORCH_CHECK(mat.size(1) == vec.size(0), "size mismatch, got input (", self.size(0), "), mat (", mat.size(0), "x",
              mat.size(1), "), vec (", vec.size(0), ")");
  at::Tensor r = addmm_any(self.dim() == 1 ? self.unsqueeze(1) : self, mat, vec.unsqueeze(1), beta, alpha).squeeze(1);
  at::native::resize_output(out, r.sizes());
  out.copy_(r);
  return out;
}

// The compressed-index invariants (the CPU's ValidateCompressedIndicesCommon.h:
// first compressed index 0, last nnz, steps in [0, dim], plain indices in
// [0, dim) and increasing within each compressed slice), checked on the GPU
// with one transfer of the verdict. Inputs the CPU rejects before its
// checks (shapes, dtypes, overflow), and any that fail one, go to the CPU,
// which raises with its message.
void validate_compressed_indices(bool is_crow, const at::Tensor& cidx, const at::Tensor& idx, int64_t cdim,
                                 int64_t dim, int64_t nnz) {
  auto cpu = [&] { on_cpu("aten::_validate_compressed_sparse_indices", "", is_crow, cidx, idx, cdim, dim, nnz); };
  at::ScalarType t = idx.scalar_type();
  bool shapes = cidx.dim() >= 1 && idx.dim() >= 1 && cidx.size(-1) == cdim + 1 && idx.size(-1) == nnz &&
                cidx.numel() / (cdim + 1) == (nnz > 0 ? idx.numel() / nnz : cidx.numel() / (cdim + 1));
  int64_t limit = t == at::kInt ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int64_t>::max();
  if (!is_vk(cidx) || !is_vk(idx) || !(t == at::kInt || t == at::kLong) || cidx.scalar_type() != t || !shapes ||
      dim > limit || cdim > limit || nnz > limit) {
    cpu();
    return;
  }
  at::Tensor c = cidx.reshape({-1, cdim + 1}).to(at::kLong);
  int64_t batches = c.size(0);
  std::vector<at::Tensor> checks;
  checks.push_back(at::eq(c.select(1, 0), 0).all());
  checks.push_back(at::eq(c.select(1, cdim), nnz).all());
  if (cdim > 0) {
    at::Tensor step = c.narrow(1, 1, cdim) - c.narrow(1, 0, cdim);
    checks.push_back(at::logical_and(at::ge(step, 0), at::le(step, dim)).all());
  }
  if (nnz > 0 && batches > 0) {
    at::Tensor p = idx.reshape({batches, nnz}).to(at::kLong);
    checks.push_back(at::logical_and(at::ge(p, 0), at::lt(p, dim)).all());
    if (nnz > 1) {
      // position j + 1 starts a slice, or the plain index grows from j to j + 1
      at::Tensor starts = at::zeros({batches, nnz + 1}, c.options().dtype(at::kBool));
      starts.scatter_(1, at::clamp(c, 0, nnz), true);
      at::Tensor grows = at::lt(p.narrow(1, 0, nnz - 1), p.narrow(1, 1, nnz - 1));
      checks.push_back(at::logical_or(starts.narrow(1, 1, nnz - 1), grows).all());
    }
  }
  if (!at::stack(checks).all().item<bool>()) cpu();
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  using namespace at::native;
  m.impl("_sparse_compressed_tensor_with_dims", TORCH_FN(sparse_compressed_tensor_with_dims));
  m.impl("empty.memory_format", TORCH_FN(empty_sparse_compressed_symint));
  m.impl("empty_like", TORCH_FN(empty_like_sparse_csr));
  m.impl("crow_indices", TORCH_FN(crow_indices_sparse_csr));
  m.impl("col_indices", TORCH_FN(col_indices_sparse_csr));
  m.impl("ccol_indices", TORCH_FN(ccol_indices_sparse_csr));
  m.impl("row_indices", TORCH_FN(row_indices_sparse_csr));
  m.impl("values", TORCH_FN(values_sparse_csr));
  m.impl("_nnz", TORCH_FN(_nnz_sparse_csr));
  m.impl("sparse_dim", TORCH_FN(sparse_dim_sparse_csr));
  m.impl("dense_dim", TORCH_FN(dense_dim_sparse_csr));
  m.impl("copy_", TORCH_FN(copy_sparse_compressed_));
  m.impl("clone", TORCH_FN(clone_sparse_compressed));
  m.impl("resize_as_sparse_", TORCH_FN(resize_as_sparse_compressed_));
  m.impl("zero_", TORCH_FN(zero_sparse_csr_));
  m.impl("_to_dense", TORCH_FN(vk::compressed_to_dense));
  m.impl("sum", TORCH_FN(sum_csr));
  // (sparse_mask: vk::sparse_mask_compressed, below)
  m.impl("select.int", TORCH_FN(select_sparse_csr));
  m.impl("mul.Tensor", TORCH_FN(mul_sparse_csr));
  m.impl("mul_.Tensor", TORCH_FN(mul_sparse_csr_));
  m.impl("mul.Scalar", TORCH_FN(mul_scalar_sparse_csr));
  m.impl("mul_.Scalar", TORCH_FN(mul__scalar_sparse_csr));
  m.impl("add.Tensor", TORCH_FN(add_sparse_csr));
  m.impl("add_.Tensor", TORCH_FN(add_sparse_csr_));
#define VK_CSR_UNARY(op)                                \
  m.impl(#op, TORCH_FN(op##_sparse_csr));               \
  m.impl(#op "_", TORCH_FN(op##_sparse_csr_));          \
  m.impl(#op ".out", TORCH_FN(op##_sparse_csr_out));
  VK_CSR_UNARY(abs)
  VK_CSR_UNARY(asin)
  VK_CSR_UNARY(asinh)
  VK_CSR_UNARY(atan)
  VK_CSR_UNARY(atanh)
  VK_CSR_UNARY(ceil)
  VK_CSR_UNARY(deg2rad)
  VK_CSR_UNARY(erf)
  VK_CSR_UNARY(erfinv)
  VK_CSR_UNARY(expm1)
  VK_CSR_UNARY(floor)
  VK_CSR_UNARY(frac)
  VK_CSR_UNARY(log1p)
  VK_CSR_UNARY(neg)
  VK_CSR_UNARY(rad2deg)
  VK_CSR_UNARY(round)
  VK_CSR_UNARY(sgn)
  VK_CSR_UNARY(sign)
  VK_CSR_UNARY(sin)
  VK_CSR_UNARY(sinh)
  VK_CSR_UNARY(sqrt)
  VK_CSR_UNARY(tan)
  VK_CSR_UNARY(tanh)
  VK_CSR_UNARY(trunc)
#undef VK_CSR_UNARY
  m.impl("relu", TORCH_FN(relu_sparse_csr));
  m.impl("relu_", TORCH_FN(relu_sparse_csr_));
  m.impl("isnan", TORCH_FN(isnan_sparse_csr));
  m.impl("isinf", TORCH_FN(isinf_sparse_csr));
  m.impl("isposinf", TORCH_FN(isposinf_sparse_csr));
  m.impl("isneginf", TORCH_FN(isneginf_sparse_csr));
  m.impl("signbit", TORCH_FN(signbit_sparse_csr));
  m.impl("addmm", TORCH_FN(vk::csr_addmm));
  m.impl("addmm.out", TORCH_FN(vk::csr_addmm_out));
  m.impl("addmv.out", TORCH_FN(vk::csr_addmv_out));
  m.impl("mm", TORCH_FN(vk::csr_mm));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_convert_indices_from_coo_to_csr", TORCH_FN(vk::convert_coo_to_csr));
  m.impl("_convert_indices_from_csr_to_coo", TORCH_FN(vk::convert_csr_to_coo));
  m.impl("_validate_compressed_sparse_indices", TORCH_FN(vk::validate_compressed_indices));
}

// ---------------------------------------------------------------- more layouts and ops

namespace vk {
namespace {

bool compressed(at::Layout l) {
  return l == at::kSparseCsr || l == at::kSparseCsc || l == at::kSparseBsr || l == at::kSparseBsc;
}

// A compressed tensor in another compressed layout, through its dense form
// (PyTorch converts between them with per-device transposition kernels),
// keeping the source's index dtype as PyTorch does.
at::Tensor dense_to_compressed(const at::Tensor& x, at::Layout layout, at::OptionalIntArrayRef blocksize,
                               std::optional<int64_t> dense_dim, const at::Tensor& specified = {});

// Where sparse `t` has specified elements (explicit zeros included), dense.
at::Tensor specified_mask(const at::Tensor& t) {
  if (t.layout() == at::kSparse) {
    return make_sparse(t, t.sparse_dim(), t.dense_dim(), t.sizes(), t._indices(), at::ones_like(t._values()), false)
        .to_dense()
        .ne(0);
  }
  bool by_col = t.layout() == at::kSparseCsc || t.layout() == at::kSparseBsc;
  at::Tensor ones = at::ones_like(t.values());
  return at::_sparse_compressed_tensor_unsafe(by_col ? t.ccol_indices() : t.crow_indices(),
                                              by_col ? t.row_indices() : t.col_indices(), ones, t.sizes(),
                                              ones.options().layout(t.layout()))
      .to_dense()
      .ne(0);
}

// `self` in a compressed layout: through dense, the blocks chosen by its
// structure (a sparse source's specified elements, explicit zeros included,
// as PyTorch's conversions) or by its nonzeros (a dense source).
at::Tensor compressed_via_dense(const at::Tensor& self, at::Layout layout, at::OptionalIntArrayRef blocksize,
                                std::optional<int64_t> dense_dim) {
  at::Tensor d = self.to_dense();
  int64_t dd = dense_dim.value_or(self.layout() == at::kStrided ? 0 : self.dense_dim());
  at::Tensor r = self.layout() == at::kStrided ? d.to_sparse(layout, blocksize, dd)
                                               : dense_to_compressed(d, layout, blocksize, dd, specified_mask(self));
  at::ScalarType it = self.layout() == at::kStrided || self.layout() == at::kSparse ? at::kLong
                       : (self.layout() == at::kSparseCsr || self.layout() == at::kSparseBsr ? self.crow_indices()
                                                                                             : self.ccol_indices())
                             .scalar_type();
  bool rows = layout == at::kSparseCsr || layout == at::kSparseBsr;
  at::Tensor ci = rows ? r.crow_indices() : r.ccol_indices();
  at::Tensor pi = rows ? r.col_indices() : r.row_indices();
  // (indices contiguous per batch, as PyTorch's checks require)
  return at::_sparse_compressed_tensor_unsafe(packed(ci.to(it)), packed(pi.to(it)), packed(r.values()),
                                              r.sizes(), r.values().options().layout(layout));
}

// The conversions PyTorch does not do, with its errors.
void check_conversion(const at::Tensor& self, at::Layout dst) {
  at::Layout src = self.layout();
  bool dst_blocked = dst == at::kSparseBsr || dst == at::kSparseBsc;
  if ((src == at::kSparseBsr || src == at::kSparseBsc) && (dst == at::kSparseCsr || dst == at::kSparseCsc)) {
    TORCH_CHECK(false, "sparse_compressed_to_sparse_", dst == at::kSparseCsr ? "csr" : "csc",
                ": expected SparseCsr or SparseCsc layout but got ", src);
  }
  if (src == at::kSparse && dst != at::kSparse) {
    TORCH_CHECK(self.sparse_dim() == 2, "sparse_coo_to_sparse: conversion from Sparse to ", dst,
                " for input tensors with sparse_dim()!=2 is not supported");
  }
  if ((src == at::kSparseCsr || src == at::kSparseCsc) && dst_blocked) {
    TORCH_CHECK(self.dim() - self.dense_dim() == 2, "sparse_compressed_to_sparse: conversion from ", src, " to ", dst,
                " for batched inputs is not supported");
  }
}

at::Tensor to_csr_any(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  check_conversion(self, at::kSparseCsr);
  if (self.layout() == at::kSparseCsr && (!dense_dim.has_value() || *dense_dim == self.dense_dim())) return self.clone();
  return compressed_via_dense(self, at::kSparseCsr, std::nullopt, dense_dim);
}
at::Tensor to_csc_any(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  check_conversion(self, at::kSparseCsc);
  if (self.layout() == at::kSparseCsc && (!dense_dim.has_value() || *dense_dim == self.dense_dim())) return self.clone();
  if (self.dim() == 2 && (!dense_dim.has_value() || *dense_dim == 0) && self.layout() != at::kSparseBsr &&
      self.layout() != at::kSparseBsc && (self.layout() != at::kSparseCsr || self.dense_dim() == 0) &&
      (self.layout() != at::kSparse || self.dense_dim() == 0)) {
    at::Tensor r = to_csc(self, dense_dim);
    if (self.layout() == at::kSparseCsr && self.crow_indices().scalar_type() == at::kInt) {
      return at::_sparse_compressed_tensor_unsafe(packed(r.ccol_indices().to(at::kInt)), packed(r.row_indices().to(at::kInt)),
                                                  r.values(), r.sizes(), r.values().options().layout(at::kSparseCsc));
    }
    return r;
  }
  return compressed_via_dense(self, at::kSparseCsc, std::nullopt, dense_dim);
}
// Between blocked layouts, the blocksize stays (PyTorch's check).
void check_blocksize(const at::Tensor& self, at::Layout dst, at::IntArrayRef blocksize) {
  if (self.layout() != at::kSparseBsr && self.layout() != at::kSparseBsc) return;
  int64_t vd = self.values().dim() - self.dense_dim();
  std::vector<int64_t> from = {self.values().size(vd - 2), self.values().size(vd - 1)};
  TORCH_CHECK(blocksize.vec() == from, "to_sparse: conversion from ", self.layout(), " to ", dst,
              " with blocksize changed from ", at::IntArrayRef(from), " to ", blocksize, " is not supported");
}

at::Tensor to_bsr_any(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
  check_conversion(self, at::kSparseBsr);
  check_blocksize(self, at::kSparseBsr, blocksize);
  return compressed_via_dense(self, at::kSparseBsr, blocksize, dense_dim);
}
at::Tensor to_bsc_any(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
  check_conversion(self, at::kSparseBsc);
  check_blocksize(self, at::kSparseBsc, blocksize);
  return compressed_via_dense(self, at::kSparseBsc, blocksize, dense_dim);
}

// index_select of a COO tensor: for each requested index, the entries at it
// along `dim` (entries sorted by that index; each request's run found by a
// sorted search, the runs concatenated in request order).
at::Tensor index_select_coo(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  dim = c10::maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
              "index_select(): Expected dtype int32/int64 for index");
  at::Tensor idx_req = index.reshape({-1}).to(at::kLong);
  int64_t size = self.size(dim);
  if (idx_req.numel() > 0) {
    TORCH_CHECK_INDEX(idx_req.min().item<int64_t>() >= -size && idx_req.max().item<int64_t>() < size,
                      "index_select(): index out of range");
  }
  idx_req = at::where(idx_req < 0, idx_req + size, idx_req);
  std::vector<int64_t> shape(self.sizes().begin(), self.sizes().end());
  shape[dim] = idx_req.numel();
  int64_t sd = self.sparse_dim();
  if (dim >= sd) {  // a dense dim: the values
    return make_sparse(self, sd, self.dense_dim(), shape, self._indices().clone(),
                       self._values().index_select(dim - sd + 1, idx_req), false);
  }
  at::Tensor indices = self._indices(), values = self._values();
  auto [sorted, order] = at::sort(indices.select(0, dim), /*stable=*/true, 0, false);
  at::Tensor lo = at::searchsorted(sorted, idx_req), hi = at::searchsorted(sorted, idx_req, false, true);
  at::Tensor counts = hi - lo;
  int64_t total = idx_req.numel() == 0 ? 0 : counts.sum().item<int64_t>();
  at::Tensor req = at::repeat_interleave(at::arange(idx_req.numel(), counts.options()), counts, 0, total);
  at::Tensor starts = at::cumsum(counts, 0) - counts;  // each run's first output position
  at::Tensor pos = at::arange(total, counts.options()) - starts.index_select(0, req) + lo.index_select(0, req);
  at::Tensor src = order.index_select(0, pos);
  at::Tensor new_indices = indices.index_select(1, src).clone();
  new_indices.select(0, dim).copy_(req);
  return make_sparse(self, sd, self.dense_dim(), shape, new_indices, values.index_select(0, src), false);
}

at::Tensor sparse_sparse_matmul(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "mat1 dtype ", a.scalar_type(), " does not match mat2 dtype ",
              b.scalar_type());
  return vk::mm(a, b).coalesce();
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  using namespace at::native;
  m.impl("_sparse_coo_tensor_with_dims", TORCH_FN(new_with_dims_sparse));
  m.impl("sum.dim_IntList", TORCH_FN(sum_sparse_coo));
  m.impl("conj_physical.out", TORCH_FN(conj_physical_out_sparse));
  m.impl("view_as_real", TORCH_FN(view_as_real_sparse));
  m.impl("view_as_complex", TORCH_FN(view_as_complex_sparse));
  m.impl("_sparse_broadcast_to", TORCH_FN(sparse_broadcast_to));
  m.impl("index_select", TORCH_FN(vk::index_select_coo));
  m.impl("_sparse_sparse_matmul", TORCH_FN(vk::sparse_sparse_matmul));
  m.impl("_to_sparse_bsr", TORCH_FN(vk::to_bsr_any));
  m.impl("_to_sparse_bsc", TORCH_FN(vk::to_bsc_any));
  m.impl("_to_sparse_csc", TORCH_FN(vk::to_csc_any));
}

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  using namespace at::native;
  m.impl("normal_", TORCH_FN(normal_sparse_csr_));
  // (conj_physical: vk::conj_physical_compressed, below: PyTorch's asserts CSR)
  m.impl("threshold_backward", TORCH_FN(threshold_backward_sparse_compressed));
  m.impl("threshold_backward.grad_input", TORCH_FN(threshold_backward_sparse_compressed_out));
  m.impl("sum.dim_IntList", TORCH_FN(sum_sparse_compressed));
  m.impl("_to_sparse_csr", TORCH_FN(vk::to_csr_any));
  m.impl("_to_sparse_csc", TORCH_FN(vk::to_csc_any));
  m.impl("_to_sparse_bsr", TORCH_FN(vk::to_bsr_any));
  m.impl("_to_sparse_bsc", TORCH_FN(vk::to_bsc_any));
}

namespace vk {
namespace {
// A compressed tensor with its indices and values contiguous (per batch, as
// PyTorch's checks require; PyTorch's device-generic conversions leave them
// as views of strided COO indices here).
at::Tensor contiguous_compressed(const at::Tensor& r) {
  bool rows = r.layout() == at::kSparseCsr || r.layout() == at::kSparseBsr;
  at::Tensor ci = rows ? r.crow_indices() : r.ccol_indices();
  at::Tensor pi = rows ? r.col_indices() : r.row_indices();
  return at::_sparse_compressed_tensor_unsafe(packed(ci), packed(pi), packed(r.values()), r.sizes(),
                                              r.values().options().layout(r.layout()));
}

// dense or COO [*batch, rows, cols, *dense] -> CSR: unbatched through COO
// (its rows compressed by convert_coo_to_csr), batched by PyTorch's code.
at::Tensor to_csr_from(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  check_conversion(self, at::kSparseCsr);
  int64_t dd = dense_dim.value_or(self.layout() == at::kSparse ? self.dense_dim() : 0);
  if (self.dim() - dd == 2) {
    at::Tensor coo = (self.layout() == at::kSparse ? self : self.to_sparse(2)).coalesce();
    if (coo.dense_dim() != dd) coo = coo.to_dense().to_sparse(2).coalesce();
    at::Tensor idx = coo._indices();
    at::Tensor crow = convert_coo_to_csr(idx.select(0, 0), self.size(0), false);
    return at::_sparse_compressed_tensor_unsafe(packed(crow), packed(idx.select(0, 1)), packed(coo._values()),
                                                self.sizes(), coo._values().options().layout(at::kSparseCsr));
  }
  return contiguous_compressed(self.layout() == at::kSparse ? at::native::coo_to_sparse_csr(self, dense_dim)
                                                            : at::native::dense_to_sparse_csr(self, dense_dim));
}

at::Tensor dense_to_csc(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  if (self.dim() == 2 && dense_dim.value_or(0) == 0) return to_csc(self, dense_dim);
  return contiguous_compressed(at::native::dense_to_sparse_csc(self, dense_dim));
}

at::Tensor dense_to_bsr(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
  return contiguous_compressed(at::native::dense_to_sparse_bsr(self, blocksize, dense_dim));
}
at::Tensor dense_to_bsc(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
  return contiguous_compressed(at::native::dense_to_sparse_bsc(self, blocksize, dense_dim));
}
}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl("_to_sparse_csr", TORCH_FN(vk::to_csr_from));
}

// ---------------------------------------------------------------- to_sparse between every layout

namespace vk {
namespace {

at::Tensor to_layout(const at::Tensor& self, std::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize,
                     std::optional<int64_t> dense_dim) {
  at::Layout l = layout.value_or(at::kSparse);
  if (l == at::kSparse) {
    if (self.layout() == at::kSparse) {
      return at::native::sparse_coo_to_sparse(self, layout, blocksize, dense_dim);
    }
    // compressed -> COO: CSR by PyTorch's code, the others from their
    // entries (explicit zeros kept), or through dense for other dense dims
    if (self.layout() == at::kSparseCsr && !blocksize.has_value()) {
      return at::native::sparse_compressed_to_sparse(self, layout, blocksize, dense_dim);
    }
    int64_t dd = dense_dim.value_or(self.dense_dim());
    if (dd == self.dense_dim()) return compressed_entries(self);
    return self.to_dense().to_sparse(self.dim() - dd);
  }
  switch (l) {
    case at::kSparseCsr: return to_csr_any(self, dense_dim);
    case at::kSparseCsc: return to_csc_any(self, dense_dim);
    default:
      TORCH_CHECK(blocksize.has_value(), "to_sparse: blocksize needed for ", l);
      return l == at::kSparseBsr ? to_bsr_any(self, *blocksize, dense_dim) : to_bsc_any(self, *blocksize, dense_dim);
  }
}

at::Tensor compressed_to_coo(const at::Tensor& self, int64_t sparse_dim) {
  if (self.layout() == at::kSparseCsr) return at::native::sparse_compressed_to_sparse(self, sparse_dim);
  if (sparse_dim == self.dim() - self.dense_dim()) return compressed_entries(self);
  return self.to_dense().to_sparse(sparse_dim);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl("_to_sparse", TORCH_FN(vk::to_layout));
}

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  m.impl("_to_sparse", TORCH_FN(vk::to_layout));
  m.impl("_to_sparse.sparse_dim", TORCH_FN(vk::compressed_to_coo));
}

// ---------------------------------------------------------------- dense -> any compressed layout

namespace vk {
namespace {

// x [*batch, M, N, *dense] as CSR/CSC/BSR/BSC with `blocksize` (R, C) (1x1
// for CSR/CSC, whose values then have no block dims): the blocks with a
// nonzero element, row-major (CSR/BSR) or column-major (CSC/BSC), each
// batch's compressed indices from the counts of its rows (columns). Every
// batch must have the same number of blocks, as PyTorch requires.
at::Tensor dense_to_compressed(const at::Tensor& x, at::Layout layout, at::OptionalIntArrayRef blocksize,
                               std::optional<int64_t> dense_dim, const at::Tensor& specified) {
  int64_t dd = dense_dim.value_or(0);
  int64_t nb = x.dim() - 2 - dd;  // batch dims
  if (nb == 0 && dd == 0 && (layout == at::kSparseCsr || layout == at::kSparseCsc) && x.dim() == 2) {
    // (one matrix: the entries in row- or column-major order from one
    // nonzero, the compressed indices by a sorted search)
    bool cols_first = layout == at::kSparseCsc;
    at::Tensor m = specified.defined() ? specified : at::ne(x, 0);
    at::Tensor nz = at::nonzero(cols_first ? m.t() : m);  // [nnz, 2]: outer, inner
    at::Tensor o = nz.select(1, 0), in = nz.select(1, 1);
    int64_t M = x.size(0), N = x.size(1);
    at::Tensor lin = cols_first ? in * N + o : o * N + in;
    at::Tensor vals = x.reshape({-1}).index_select(0, lin);
    at::Tensor compressed = at::searchsorted(o.contiguous(), at::arange((cols_first ? N : M) + 1, o.options()));
    return at::_sparse_compressed_tensor_unsafe(packed(compressed), packed(in), packed(vals), x.sizes(),
                                                x.options().layout(layout));
  }
  TORCH_CHECK(nb >= 0, "to_sparse: expected a tensor with at least ", 2 + dd, " dims, got ", x.dim());
  bool blocked = layout == at::kSparseBsr || layout == at::kSparseBsc;
  bool by_col = layout == at::kSparseCsc || layout == at::kSparseBsc;
  int64_t R = 1, C = 1;
  if (blocked) {
    TORCH_CHECK(blocksize.has_value() && blocksize->size() == 2, "to_sparse: blocksize needs 2 elements for ", layout);
    R = (*blocksize)[0];
    C = (*blocksize)[1];
  }
  int64_t M = x.size(nb), N = x.size(nb + 1);
  TORCH_CHECK(R > 0 && C > 0 && M % R == 0 && N % C == 0, "to_sparse: tensor sizes (", M, ", ", N,
              ") must be divisible by the blocksize (", R, ", ", C, ")");
  int64_t MB = M / R, NB = N / C;
  std::vector<int64_t> bshape(x.sizes().begin(), x.sizes().begin() + nb);
  std::vector<int64_t> dshape(x.sizes().begin() + nb + 2, x.sizes().end());
  int64_t B = 1, D = 1;
  for (int64_t s : bshape) B *= s;
  for (int64_t s : dshape) D *= s;
  // [B, MB, R, NB, C, D] -> [B, MB, NB, R, C, D]
  std::vector<int64_t> v = {B, MB, R, NB, C, D};
  at::Tensor blocks = x.reshape(v).permute({0, 1, 3, 2, 4, 5});
  // the blocks with a nonzero (or, given `specified`, a specified) element
  at::Tensor mask = (specified.defined() ? specified.reshape(v).permute({0, 1, 3, 2, 4, 5}) : at::ne(blocks, 0))
                        .flatten(3)
                        .any(3);  // [B, MB, NB]
  if (by_col) {
    blocks = blocks.transpose(1, 2);
    mask = mask.transpose(1, 2);
  }
  int64_t outer = by_col ? NB : MB;
  at::Tensor nz = at::nonzero(mask);  // [nnz, 3]: batch, outer, inner (sorted)
  int64_t nnz = nz.size(0);
  TORCH_CHECK(B == 0 || nnz % std::max<int64_t>(B, 1) == 0, "Expect the same number of specified elements per batch.");
  at::TensorOptions lopts = x.options().dtype(at::kLong);
  at::Tensor b = nz.select(1, 0), o = nz.select(1, 1), in = nz.select(1, 2);
  at::Tensor counts = at::zeros({B * outer}, lopts).index_add_(0, b * outer + o, at::ones({nnz}, lopts));
  at::Tensor per_batch = counts.view({B, outer}).sum(1);
  if (B > 1) {
    TORCH_CHECK(at::eq(per_batch, per_batch[0]).all().item<bool>(), "Expect the same number of specified elements per batch.");
  }
  at::Tensor compressed = at::cat({at::zeros({B, 1}, lopts), at::cumsum(counts.view({B, outer}), 1)}, 1);
  int64_t per = B == 0 ? 0 : nnz / B;
  at::Tensor vals = blocks.reshape({B * MB * NB, R, C, D})
                        .index_select(0, (b * outer + o) * (by_col ? MB : NB) + in);  // [nnz, R, C, D]
  std::vector<int64_t> vshape = bshape;
  vshape.push_back(per);
  if (blocked) {
    vshape.push_back(R);
    vshape.push_back(C);
  }
  vshape.insert(vshape.end(), dshape.begin(), dshape.end());
  std::vector<int64_t> cshape = bshape, pshape = bshape;
  cshape.push_back(outer + 1);
  pshape.push_back(per);
  return at::_sparse_compressed_tensor_unsafe(packed(compressed.reshape(cshape)), packed(in.reshape(pshape)),
                                              packed(vals.reshape(vshape)), x.sizes(),
                                              x.options().layout(layout));
}

// sparse_mask(dense, compressed mask): the dense tensor's blocks where the
// mask has its (specified, zero or not) blocks, as PyTorch's.
at::Tensor sparse_mask_compressed(const at::Tensor& self, const at::Tensor& mask) {
  if (self.layout() != at::kStrided || self.is_same(mask) || !mask.numel() || !mask._nnz()) {
    return at::native::sparse_mask_sparse_compressed(self, mask);
  }
  TORCH_CHECK(mask.sizes().equals(self.sizes()), "sparse_mask(): operands have incompatible sizes; self has size ",
              self.sizes(), " but mask has size ", mask.sizes());
  at::Layout l = mask.layout();
  bool by_col = l == at::kSparseCsc || l == at::kSparseBsc;
  at::Tensor ci = by_col ? mask.ccol_indices() : mask.crow_indices();
  at::Tensor pi = by_col ? mask.row_indices() : mask.col_indices();
  if ((l == at::kSparseCsr || l == at::kSparseCsc) && mask.dim() == 2 && mask.dense_dim() == 0) {
    // (one matrix: the dense values at the mask's entries, its indices kept)
    at::Tensor c = ci.to(at::kLong), in = pi.to(at::kLong);
    at::Tensor o = at::searchsorted(c, at::arange(in.numel(), c.options()), false, true) - 1;
    int64_t N = self.size(1);
    at::Tensor lin = by_col ? in * N + o : o * N + in;
    at::Tensor vals = self.reshape({-1}).index_select(0, lin);
    return at::_sparse_compressed_tensor_unsafe(ci.clone(), pi.clone(), vals, self.sizes(), vals.options().layout(l));
  }
  at::Tensor ones = at::ones(mask.values().sizes(), self.options().dtype(at::kBool));
  at::Tensor specified =
      at::_sparse_compressed_tensor_unsafe(ci, pi, ones, self.sizes(), ones.options().layout(l)).to_dense();
  std::optional<std::vector<int64_t>> bs;
  if (l == at::kSparseBsr || l == at::kSparseBsc) {
    bs = std::vector<int64_t>{mask.values().size(mask.dim() - mask.dense_dim() - 1),
                              mask.values().size(mask.dim() - mask.dense_dim())};
  }
  at::Tensor r = dense_to_compressed(self, l, bs.has_value() ? at::OptionalIntArrayRef(*bs) : std::nullopt,
                                     mask.dense_dim(), specified);
  if (ci.scalar_type() == at::kInt) {
    at::Tensor rc = by_col ? r.ccol_indices() : r.crow_indices(), rp = by_col ? r.row_indices() : r.col_indices();
    return at::_sparse_compressed_tensor_unsafe(packed(rc.to(at::kInt)), packed(rp.to(at::kInt)), r.values(),
                                                r.sizes(), r.values().options().layout(l));
  }
  return r;
}

at::Tensor dense_csr(const at::Tensor& x, std::optional<int64_t> dd) {
  return dense_to_compressed(x, at::kSparseCsr, std::nullopt, dd);
}
at::Tensor dense_csc(const at::Tensor& x, std::optional<int64_t> dd) {
  return dense_to_compressed(x, at::kSparseCsc, std::nullopt, dd);
}
at::Tensor dense_bsr(const at::Tensor& x, at::IntArrayRef bs, std::optional<int64_t> dd) {
  return dense_to_compressed(x, at::kSparseBsr, bs, dd);
}
at::Tensor dense_bsc(const at::Tensor& x, at::IntArrayRef bs, std::optional<int64_t> dd) {
  return dense_to_compressed(x, at::kSparseBsc, bs, dd);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_to_sparse_csr", TORCH_FN(vk::dense_csr));
  m.impl("_to_sparse_csc", TORCH_FN(vk::dense_csc));
  m.impl("_to_sparse_bsr", TORCH_FN(vk::dense_bsr));
  m.impl("_to_sparse_bsc", TORCH_FN(vk::dense_bsc));
}

// ---------------------------------------------------------------- more compressed ops

namespace vk {
namespace {

// sum of a CSR tensor over dims (keepdim, as PyTorch's sparse CSR sum): the
// dense sum, back to CSR.
// A CSR tensor reduced over dims (sum, or prod: over the specified elements
// only, unspecified ones count as zeros for sum and as ones for prod, as
// PyTorch's masked reductions): CSR where any reduced element was specified.
// The products of v's rows [n, *] by key into out (ones) [groups, *]: the
// rows sorted by key, then neighbours of a key multiplied pairwise, halving
// the rows per round (log2 of the longest run rounds).
at::Tensor segment_prod(const at::Tensor& key, const at::Tensor& v0, at::Tensor out) {
  auto [ks, order] = at::sort(key, /*stable=*/true, 0, false);
  at::Tensor v = v0.index_select(0, order);
  while (ks.numel() > 1) {
    int64_t n = ks.numel();
    at::Tensor same_next = at::eq(ks.narrow(0, 1, n - 1), ks.narrow(0, 0, n - 1));  // [n - 1]
    if (!same_next.any().item<bool>()) break;  // (one row per key)
    at::Tensor head = at::cat({at::ones({1}, same_next.options()), at::logical_not(same_next)});
    at::Tensor idx = at::arange(n, ks.options());
    at::Tensor start = std::get<0>(at::cummax(at::where(head, idx, at::zeros({}, idx.options())), 0));
    at::Tensor even = at::eq(at::remainder(idx - start, 2), 0);
    at::Tensor has_next = at::cat({same_next, at::zeros({1}, same_next.options())});
    std::vector<int64_t> one_shape(v.sizes().begin(), v.sizes().end());
    one_shape[0] = 1;
    at::Tensor next_v = at::cat({v.narrow(0, 1, n - 1), at::ones(one_shape, v.options())});
    std::vector<int64_t> bshape(v.dim(), 1);
    bshape[0] = n;
    at::Tensor partner = at::where(has_next.view(bshape), next_v, at::ones({}, v.options()));
    at::Tensor keep = at::nonzero(even).reshape({-1});
    v = (v * partner).index_select(0, keep);
    ks = ks.index_select(0, keep);
  }
  return out.index_copy_(0, ks, v);
}

at::Tensor csr_reduce(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype,
                      bool prod) {
  // From the entries, not through dense: the reduced result (its keepdim
  // shape) accumulates them by their kept indices, so memory grows with nnz
  // and the result's size.
  TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
  int64_t nd = self.dim();
  std::vector<bool> reduced(nd, false);
  for (int64_t k : dim) reduced[at::maybe_wrap_dim(k, nd)] = true;
  if (dim.empty() && !prod) std::fill(reduced.begin(), reduced.end(), true);  // (at::sum over no dims: all)
  at::ScalarType st = self.scalar_type();
  at::ScalarType out_t = dtype.value_or(at::isIntegralType(st, true) ? at::kLong : st);
  at::Tensor e = compressed_entries(self);
  int64_t sd = e.sparse_dim();
  at::Tensor vals = e._values().to(out_t);  // [nnz, *dense]
  for (int64_t d = nd - 1; d >= sd; --d) {
    if (!reduced[d]) continue;
    vals = prod ? at::prod(vals, d - sd + 1, true) : at::sum(vals, {d - sd + 1}, true);
  }
  std::vector<int64_t> kd_shape(self.sizes().begin(), self.sizes().end());
  for (int64_t d = 0; d < nd; ++d) {
    if (reduced[d]) kd_shape[d] = 1;
  }
  std::vector<int64_t> kd_sparse(kd_shape.begin(), kd_shape.begin() + sd);
  int64_t groups = 1;
  for (int64_t s : kd_sparse) groups *= s;
  at::Tensor idx = e._indices().clone();
  for (int64_t d = 0; d < sd; ++d) {
    if (reduced[d]) idx.select(0, d).zero_();
  }
  std::vector<int64_t> flat_shape = {groups};
  flat_shape.insert(flat_shape.end(), vals.sizes().begin() + 1, vals.sizes().end());
  at::Tensor r = prod ? at::ones(flat_shape, vals.options()) : at::zeros(flat_shape, vals.options());
  at::Tensor any = at::zeros(flat_shape, vals.options().dtype(at::kBool));
  if (e._nnz() > 0) {
    at::Tensor key = linear_keys(idx, kd_sparse, sd);
    if (prod && at::isComplexType(out_t)) {
      r = segment_prod(key, vals, r);  // (index_reduce has no complex kernel)
    } else {
      r = prod ? r.index_reduce_(0, key, vals, "prod", true) : r.index_add_(0, key, vals);
    }
    any.index_fill_(0, key, true);
  }
  r = r.view(kd_shape);
  any = any.view(kd_shape);
  if (!keepdim) {
    for (int64_t d = nd - 1; d >= 0; --d) {
      if (reduced[d]) {
        r = r.squeeze(d);
        any = any.squeeze(d);
      }
    }
  }
  if (r.dim() < 2) return r.contiguous();
  return to_csr_from(dense_to_compressed(r.contiguous(), at::kSparseCsr, std::nullopt, 0, any.contiguous()), std::nullopt);
}

at::Tensor csr_sum_dim(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  return csr_reduce(self, dim, keepdim, dtype, false);
}

at::Tensor csr_prod_dim(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  return csr_reduce(self, dim, keepdim, dtype, true);
}

// A triangular system with a compressed matrix A: solved with A dense.
std::tuple<at::Tensor&, at::Tensor&> triangular_solve_compressed_out(const at::Tensor& self, const at::Tensor& A,
                                                                      bool upper, bool transpose, bool unitriangular,
                                                                      at::Tensor& X, at::Tensor& M) {
  auto [x, m] = at::triangular_solve(self, A.to_dense(), upper, transpose, unitriangular);
  at::native::resize_output(X, x.sizes());
  if (A._nnz() == 0) {
    X.fill_(std::numeric_limits<double>::quiet_NaN());  // (as PyTorch's sparse solve: no coefficients)
  } else {
    X.copy_(x);
  }
  return {X, M};
}

// CSR (+) CSR or dense (+) CSR: the entries of both (through COO, explicit
// zeros kept), or the dense sum.
at::Tensor& csr_add_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
  if (self.layout() == at::kStrided || other.layout() == at::kStrided || out.layout() == at::kStrided) {
    // (compressed + dense: dense + alpha * compressed, as PyTorch's)
    bool swap = self.layout() != at::kStrided && other.layout() == at::kStrided;
    at::Tensor r = swap ? at::add(other, dense_of(self), alpha) : at::add(dense_of(self), dense_of(other), alpha);
    if (compressed_layout(out)) {
      assign_compressed(out, to_layout(r, out.layout(), std::nullopt, out.dense_dim()));
      return out;
    }
    if (!out.is_same(self)) at::native::resize_output(out, r.sizes());
    out.copy_(r);
    return out;
  }
  TORCH_CHECK(self.sizes() == other.sizes(), "Expected input tensors to have the same shape, but got tensor(", self.sizes(),
              ") and tensor(", other.sizes(), ")");
  at::Tensor a = self.to_sparse(), b = other.to_sparse();
  at::Tensor r = to_layout(at::add(a, b, alpha).coalesce(), self.layout(), std::nullopt, self.dense_dim());
  assign_compressed(out, r);
  return out;
}

// conj_physical of any compressed layout: the values conjugated.
at::Tensor conj_physical_compressed(const at::Tensor& self) {
  bool by_col = self.layout() == at::kSparseCsc || self.layout() == at::kSparseBsc;
  at::Tensor values = self.is_complex() ? self.values().conj_physical() : self.values().clone();
  return at::_sparse_compressed_tensor_unsafe((by_col ? self.ccol_indices() : self.crow_indices()).clone(),
                                              (by_col ? self.row_indices() : self.col_indices()).clone(), values,
                                              self.sizes(), values.options().layout(self.layout()));
}
at::Tensor& conj_physical_compressed_(at::Tensor& self) {
  if (self.is_complex()) self.values().conj_physical_();
  return self;
}
at::Tensor& conj_physical_compressed_out(const at::Tensor& self, at::Tensor& out) {
  assign_compressed(out, conj_physical_compressed(self));
  return out;
}

// beta self + alpha (mat1 @ mat2) at the entries of the CSR self.
at::Tensor sampled_addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                         const at::Scalar& alpha) {
  TORCH_CHECK(self.layout() == at::kSparseCsr && self.dense_dim() == 0, "sampled_addmm: expected a CSR self");
  at::Tensor product = at::matmul(dense_of(mat1), dense_of(mat2));
  at::Tensor c = self;
  if (product.dim() > self.dim()) {  // self broadcast over the product's batch dims
    std::vector<int64_t> batch(product.sizes().begin(), product.sizes().end() - 2);
    auto over_batch = [&](const at::Tensor& t) {
      std::vector<int64_t> shape = batch;
      shape.push_back(t.size(-1));
      return packed(t.expand(shape).contiguous());
    };
    std::vector<int64_t> size = batch;
    size.push_back(self.size(-2));
    size.push_back(self.size(-1));
    c = at::_sparse_compressed_tensor_unsafe(over_batch(self.crow_indices()), over_batch(self.col_indices()),
                                             over_batch(self.values()), size, self.values().options().layout(at::kSparseCsr));
  }
  at::Tensor idx = compressed_entries(c)._indices();  // (batch..., row, col), in storage order
  c10::List<std::optional<at::Tensor>> where;
  for (int64_t d = 0; d < idx.size(0); ++d) where.push_back(idx.select(0, d));
  at::Tensor picked = at::index(product.expand(c.sizes()), where).reshape(c.values().sizes());
  at::Tensor values = at::add(c.values() * beta, picked, alpha);
  return at::_sparse_compressed_tensor_unsafe(packed(c.crow_indices()).clone(), packed(c.col_indices()).clone(), values,
                                              c.sizes(), values.options().layout(at::kSparseCsr));
}

at::Tensor& sampled_addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                              const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  assign_compressed(out, sampled_addmm(self, mat1, mat2, beta, alpha));
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  using namespace at::native;
  m.impl("angle", TORCH_FN(angle_sparse_csr));
  m.impl("angle.out", TORCH_FN(angle_sparse_csr_out));
  m.impl("mul.out", TORCH_FN(mul_out_sparse_csr));
  m.impl("isneginf.out", TORCH_FN(isneginf_sparse_csr_out));
  m.impl("isposinf.out", TORCH_FN(isposinf_sparse_csr_out));
  m.impl("signbit.out", TORCH_FN(signbit_sparse_csr_out));
  m.impl("resize_", TORCH_FN(resize_sparse_csr_));
  m.impl("fill_.Scalar", TORCH_FN(fill_sparse_csr_));
  m.impl("select_copy.int", TORCH_FN(select_copy_sparse_csr));
  m.impl("_sparse_csr_sum.dim_dtype", TORCH_FN(vk::csr_sum_dim));
  m.impl("_sparse_csr_prod.dim_dtype", TORCH_FN(vk::csr_prod_dim));
  m.impl("triangular_solve.X", TORCH_FN(vk::triangular_solve_compressed_out));
  m.impl("add.out", TORCH_FN(vk::csr_add_out));
  m.impl("sparse_mask", TORCH_FN(vk::sparse_mask_compressed));
  m.impl("conj_physical", TORCH_FN(vk::conj_physical_compressed));
  m.impl("conj_physical_", TORCH_FN(vk::conj_physical_compressed_));
  m.impl("conj_physical.out", TORCH_FN(vk::conj_physical_compressed_out));
  m.impl("sparse_sampled_addmm", TORCH_FN(vk::sampled_addmm));
  m.impl("sparse_sampled_addmm.out", TORCH_FN(vk::sampled_addmm_out));
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  using namespace at::native;
  m.impl("isneginf.out", TORCH_FN(isneginf_sparse_out));
  m.impl("isposinf.out", TORCH_FN(isposinf_sparse_out));
  m.impl("signbit.out", TORCH_FN(signbit_sparse_out));
}

// ---------------------------------------------------------------- more COO ops

namespace vk {
namespace {

// The pool of each entry of a coalesced COO tensor for a reduction along
// sparse dim `dim`: the entries with the same indices in the other sparse
// dims share one. Returns (pool per entry, pool count).
std::pair<at::Tensor, int64_t> sparse_pools(const at::Tensor& c, int64_t dim) {
  at::Tensor idx = c._indices().clone();
  idx.select(0, dim).zero_();
  at::Tensor key = linear_keys(idx, c.sizes(), c.sparse_dim());
  auto [uniq, inverse, counts] = at::_unique2(key, true, true, false);
  return {inverse.reshape({-1}), uniq.numel()};
}

// v [nnz, *dense] summed over each pool, back at every entry.
at::Tensor pool_sum(const at::Tensor& v, const at::Tensor& pool, int64_t pools) {
  std::vector<int64_t> shape(v.sizes().begin(), v.sizes().end());
  shape[0] = pools;
  return at::zeros(shape, v.options()).index_add_(0, pool, v).index_select(0, pool);
}

// v's leading (entry) dim mask `m` [nnz], broadcastable against v.
at::Tensor per_entry(const at::Tensor& m, const at::Tensor& v) {
  std::vector<int64_t> shape(v.dim(), 1);
  shape[0] = v.size(0);
  return m.view(shape);
}

// softmax / log_softmax of a COO tensor along `dim`, over its specified
// elements only (the unspecified ones count as -inf), as PyTorch's.
at::Tensor sparse_softmax_impl(const at::Tensor& self, int64_t dim, bool half_to_float, bool log) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  at::Tensor c = self.coalesce();
  int64_t sd = c.sparse_dim();
  at::Tensor v = c._values();
  if (half_to_float) v = v.to(at::kFloat);
  at::Tensor out;
  if (c._nnz() == 0) {
    out = v.clone();
  } else if (dim >= sd) {
    out = log ? at::log_softmax(v, dim - sd + 1) : at::softmax(v, dim - sd + 1);
  } else {
    auto [pool, pools] = sparse_pools(c, dim);
    std::vector<int64_t> shape(v.sizes().begin(), v.sizes().end());
    shape[0] = pools;
    at::Tensor gi = per_entry(pool, v).expand_as(v);
    at::Tensor mx = at::zeros(shape, v.options()).scatter_reduce(0, gi, v, "amax", false).index_select(0, pool);
    at::Tensor shifted = v - mx;
    at::Tensor e = at::exp(shifted);
    at::Tensor sum = pool_sum(e, pool, pools);
    out = log ? shifted - at::log(sum) : e / sum;
  }
  return make_sparse(c, sd, c.dense_dim(), c.sizes(), c._indices().clone(), out, true);
}

at::Tensor sparse_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return sparse_softmax_impl(self, dim, half_to_float, false);
}
at::Tensor sparse_log_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return sparse_softmax_impl(self, dim, half_to_float, true);
}

// The values of COO `g0` at the entries of coalesced `like` (zero where
// `g0` has none).
at::Tensor values_at(const at::Tensor& g0, const at::Tensor& like) {
  at::Tensor g = g0.coalesce();
  int64_t sd = like.sparse_dim();
  if (g._nnz() == 0) return at::zeros_like(like._values(), g._values().options());
  at::Tensor kg = linear_keys(g._indices(), g.sizes(), sd), kl = linear_keys(like._indices(), like.sizes(), sd);
  at::Tensor pos = at::searchsorted(kg, kl).clamp_max(kg.numel() - 1);
  at::Tensor hit = at::eq(kg.index_select(0, pos), kl);
  at::Tensor v = g._values().index_select(0, pos);
  return at::where(per_entry(hit, v), v, at::zeros({}, v.options()));
}

at::Tensor sparse_softmax_backward_impl(const at::Tensor& grad, const at::Tensor& output0, int64_t dim, bool log) {
  dim = at::maybe_wrap_dim(dim, output0.dim());
  at::Tensor output = output0.coalesce();
  int64_t sd = output.sparse_dim();
  at::Tensor out = output._values();
  at::Tensor g = values_at(grad, output).to(out.scalar_type());
  at::Tensor r;
  if (output._nnz() == 0) {
    r = out.clone();
  } else if (dim >= sd) {
    int64_t d = dim - sd + 1;
    r = log ? g - at::exp(out) * g.sum(d, true) : out * (g - (g * out).sum(d, true));
  } else {
    auto [pool, pools] = sparse_pools(output, dim);
    r = log ? g - at::exp(out) * pool_sum(g, pool, pools) : out * (g - pool_sum(g * out, pool, pools));
  }
  return make_sparse(output, sd, output.dense_dim(), output.sizes(), output._indices().clone(), r, true);
}

at::Tensor sparse_softmax_backward(const at::Tensor& grad, const at::Tensor& output, int64_t dim, const at::Tensor&) {
  return sparse_softmax_backward_impl(grad, output, dim, false);
}
at::Tensor sparse_log_softmax_backward(const at::Tensor& grad, const at::Tensor& output, int64_t dim,
                                       const at::Tensor&) {
  return sparse_softmax_backward_impl(grad, output, dim, true);
}

// The gradient of sparse sum over `dims`: the incoming gradient (dense when
// every sparse dim was summed, else sparse) at each entry of the input,
// expanded over the summed dense dims, as PyTorch's.
at::Tensor sparse_sum_backward(const at::Tensor& grad, const at::Tensor& input0, at::IntArrayRef dims) {
  if ((grad.is_sparse() && !grad._nnz()) || !grad.numel()) return at::zeros_like(input0);
  at::Tensor input = input0.coalesce();
  int64_t nd = input.dim(), sd = input.sparse_dim();
  std::vector<bool> summed(nd, false);
  for (int64_t d : dims) summed[at::maybe_wrap_dim(d, nd)] = true;
  std::vector<int64_t> keep_sparse, sum_dense;  // (summed dense dims as values dims)
  for (int64_t d = 0; d < nd; ++d) {
    if (d < sd && !summed[d]) keep_sparse.push_back(d);
    if (d >= sd && summed[d]) sum_dense.push_back(d - sd + 1);
  }
  at::Tensor iv = input._values();
  at::Tensor gv;  // [nnz, *kept dense dims]
  if (keep_sparse.empty()) {
    TORCH_CHECK(!grad.is_sparse(), "_sparse_sum_backward: expected grad to be dense since all sparse dims are summed");
    std::vector<int64_t> es = {iv.size(0)};
    es.insert(es.end(), grad.sizes().begin(), grad.sizes().end());
    gv = grad.unsqueeze(0).expand(es);
  } else {
    TORCH_CHECK(grad.is_sparse(), "_sparse_sum_backward: expected grad to be sparse, but got dense");
    // grad's values at the input's indices in the kept sparse dims
    at::Tensor kept = input._indices().index_select(0, at::tensor(keep_sparse, input._indices().options()));
    std::vector<int64_t> kept_sizes;
    for (int64_t d : keep_sparse) kept_sizes.push_back(input.size(d));
    at::Tensor g = grad.coalesce();
    at::Tensor kg = linear_keys(g._indices(), g.sizes(), g.sparse_dim());
    at::Tensor ki = linear_keys(kept, kept_sizes, static_cast<int64_t>(keep_sparse.size()));
    at::Tensor pos = at::searchsorted(kg, ki).clamp_max(kg.numel() - 1);
    at::Tensor v = g._values().index_select(0, pos);
    gv = at::where(per_entry(at::eq(kg.index_select(0, pos), ki), v), v, at::zeros({}, v.options()));
  }
  for (int64_t d : sum_dense) gv = gv.unsqueeze(d);
  gv = gv.expand(iv.sizes()).contiguous();
  return make_sparse(input, sd, input.dense_dim(), input.sizes(), input._indices().clone(), gv.to(grad.scalar_type()),
                     input.is_coalesced());
}

// `t` (its own entries) with the values zeroed where `mask` has no entry, a
// value counted once per matching mask entry when accumulating.
at::Tensor sparse_mask_projection(const at::Tensor& t, const at::Tensor& mask, bool accumulate_matches) {
  TORCH_CHECK(mask.sizes().equals(t.sizes()), "_sparse_mask_projection(): operands have incompatible sizes; self has size ",
              t.sizes(), " but mask has size ", mask.sizes());
  if (!t.numel() || !t._nnz() || !mask._nnz()) {
    at::Tensor r = t.clone();
    r._values().zero_();
    return r;
  }
  int64_t sd = t.sparse_dim();
  at::Tensor kt = linear_keys(t._indices(), t.sizes(), sd);
  at::Tensor km = std::get<0>(at::sort(linear_keys(mask._indices(), mask.sizes(), sd)));
  at::Tensor count = at::searchsorted(km, kt, false, true) - at::searchsorted(km, kt);
  at::Tensor w = accumulate_matches ? count : at::gt(count, 0);
  at::Tensor v = t._values();
  return make_sparse(t, sd, t.dense_dim(), t.sizes(), t._indices().clone(), v * per_entry(w.to(v.scalar_type()), v),
                     t.is_coalesced());
}

// sparse [B, n, m] @ dense [B, m, p]: one block-diagonal sparse product.
at::Tensor bmm_sparse(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");
  TORCH_CHECK(self.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self.dense_dim());
  TORCH_CHECK(self.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self.sparse_dim());
  TORCH_CHECK(mat2.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2.dim());
  TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
  TORCH_CHECK(self.size(2) == mat2.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");
  int64_t B = self.size(0), n = self.size(1), m = self.size(2), p = mat2.size(2);
  at::Tensor idx = self._indices();
  at::Tensor rows = idx.select(0, 0) * n + idx.select(0, 1), cols = idx.select(0, 0) * m + idx.select(0, 2);
  at::Tensor flat = make_sparse(self, 2, 0, {B * n, B * m}, at::stack({rows, cols}), self._values(), false);
  return vk::mm(flat, mat2.reshape({B * m, p})).view({B, n, p});
}

at::Tensor& bmm_sparse_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::Tensor r = bmm_sparse(self, mat2);
  at::native::resize_output(out, r.sizes());
  out.copy_(r);
  return out;
}

// sparse [m, k] @ dense [k, n] as a hybrid COO tensor: the rows of the
// product where the sparse matrix has entries.
at::Tensor hspmm_sparse(const at::Tensor& sparse, const at::Tensor& dense) {
  TORCH_CHECK(sparse.sparse_dim() == 2, "hspmm: Argument #2: matrices expected, got ", sparse.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse.dense_dim() == 0, "hspmm: Argument #2: scalar values expected, got ", sparse.dense_dim(),
              "D values");
  TORCH_CHECK(dense.dim() == 2, "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");
  TORCH_CHECK(dense.size(0) == sparse.size(1), "hspmm: Argument #3: Expected dim 0 size ", sparse.size(1), ", got ",
              dense.size(0));
  int64_t m = sparse.size(0), n = dense.size(1);
  at::Tensor c = sparse.coalesce();
  at::Tensor rows = std::get<0>(at::_unique2(c._indices().select(0, 0), true, false, false));
  at::Tensor product = vk::mm(c, dense);
  return make_sparse(c, 1, 1, {m, n}, rows.unsqueeze(0), product.index_select(0, rows), true);
}

at::Tensor& hspmm_out(const at::Tensor& sparse, const at::Tensor& dense, at::Tensor& out) {
  out.copy_(hspmm_sparse(sparse, dense));
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl("_sparse_softmax", TORCH_FN(vk::sparse_softmax));
  m.impl("_sparse_log_softmax", TORCH_FN(vk::sparse_log_softmax));
  m.impl("_sparse_softmax_backward_data", TORCH_FN(vk::sparse_softmax_backward));
  m.impl("_sparse_log_softmax_backward_data", TORCH_FN(vk::sparse_log_softmax_backward));
  m.impl("_sparse_sum_backward", TORCH_FN(vk::sparse_sum_backward));
  m.impl("_sparse_mask_projection", TORCH_FN(vk::sparse_mask_projection));
  m.impl("bmm", TORCH_FN(vk::bmm_sparse));
  m.impl("bmm.out", TORCH_FN(vk::bmm_sparse_out));
  m.impl("hspmm", TORCH_FN(vk::hspmm_sparse));
  m.impl("hspmm.out", TORCH_FN(vk::hspmm_out));
  m.impl("linalg_vector_norm", TORCH_FN(at::native::linalg_vector_norm_sparse));
}

// ---------------------------------------------------------------- dispatch stubs

// PyTorch's flatten_indices (its COO constructor's coalesced check, among
// others) runs through a DispatchStub, declared in a header it does not ship.
namespace at::native {
using flatten_indices_fn = Tensor (*)(const Tensor& indices, IntArrayRef size);
DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub)
}  // namespace at::native

namespace vk {
namespace {
at::Tensor flatten_indices_vk(const at::Tensor& indices, at::IntArrayRef size) {
  return linear_keys(indices, size, indices.size(0));
}
}  // namespace
}  // namespace vk

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(flatten_indices_stub, &vk::flatten_indices_vk)
}  // namespace at::native
