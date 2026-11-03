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

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace vk {
namespace {

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
  auto [sorted, order] = at::sort(key, /*stable=*/true, 0, false);
  at::Tensor head = at::ones({nnz}, key.options().dtype(at::kBool));
  head.narrow(0, 1, nnz - 1).copy_(at::ne(sorted.narrow(0, 1, nnz - 1), sorted.narrow(0, 0, nnz - 1)));
  at::Tensor group = at::cumsum(head, 0, at::kLong) - 1;
  at::Tensor unique_keys = at::masked_select(sorted, head);
  int64_t u = unique_keys.numel();
  std::vector<int64_t> vshape(values.sizes().begin(), values.sizes().end());
  vshape[0] = u;
  // (summed in index order within each run, as the CPU's coalesce)
  at::Tensor summed = at::zeros(vshape, values.options()).index_add_(0, group, values.index_select(0, order));
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
  TORCH_CHECK(self.sparse_dim() == other.sparse_dim(), "add: expected 'self' and 'other' to have same sparse dimensions");
  at::Tensor ov = other._values().to(t);
  if (!(alpha.isIntegral(true) && alpha.toLong() == 1)) ov = ov * alpha;
  at::Tensor indices = at::cat({self._indices(), other._indices()}, 1);
  at::Tensor values = at::cat({self._values().to(t), ov}, 0);
  at::Tensor r = make_sparse(self, self.sparse_dim(), self.dense_dim(), self.sizes(), indices, values, false);
  if (self._nnz() == 0 || other._nnz() == 0) r._coalesced_(self._nnz() == 0 ? other.is_coalesced() : self.is_coalesced());
  out.copy_(r);
  out._coalesced_(r.is_coalesced());
  return out;
}

// sparse * sparse: the entries at the keys both have (after coalescing).
at::Tensor mul_sparse_sparse(const at::Tensor& a0, const at::Tensor& b0) {
  TORCH_CHECK(a0.sizes() == b0.sizes(), "mul operands have incompatible sizes");
  at::ScalarType t = at::result_type(a0, b0);
  at::Tensor a = a0.coalesce(), b = b0.coalesce();
  int64_t sd = a.sparse_dim();
  at::Tensor ka = linear_keys(a._indices(), a.sizes(), sd), kb = linear_keys(b._indices(), b.sizes(), sd);
  std::vector<int64_t> vshape(a._values().sizes().begin(), a._values().sizes().end());
  if (ka.numel() == 0 || kb.numel() == 0) {
    vshape[0] = 0;
    return make_sparse(a, sd, a.dense_dim(), a.sizes(), at::empty({sd, 0}, ka.options()),
                       at::empty(vshape, a._values().options().dtype(t)), true);
  }
  at::Tensor pos = at::searchsorted(kb, ka).clamp_max(kb.numel() - 1);
  at::Tensor hit = at::eq(kb.index_select(0, pos), ka);
  at::Tensor ia = at::nonzero(hit).reshape({-1});
  at::Tensor ib = pos.index_select(0, ia);
  at::Tensor values = a._values().index_select(0, ia).to(t) * b._values().index_select(0, ib).to(t);
  return make_sparse(a, sd, a.dense_dim(), a.sizes(), a._indices().index_select(1, ia), values, true);
}

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  at::ScalarType t = at::result_type(self, other);
  at::Tensor r;
  if (self.is_sparse() && other.is_sparse()) {
    r = mul_sparse_sparse(self, other);
  } else {
    const at::Tensor& s = self.is_sparse() ? self : other;
    const at::Tensor& d = self.is_sparse() ? other : self;
    if (d.dim() == 0) {  // a scalar
      r = make_sparse(s, s.sparse_dim(), s.dense_dim(), s.sizes(), s._indices().clone(), at::mul(s._values(), d).to(t),
                      s.is_coalesced());
    } else {  // sparse * dense of the same shape: sparse, the dense values at its entries
      TORCH_CHECK(d.sizes() == s.sizes(), "mul(sparse, dense): expected the same sizes, got ", s.sizes(), " and ",
                  d.sizes());
      at::Tensor c = s.coalesce();
      c10::List<std::optional<at::Tensor>> rows;
      for (int64_t k = 0; k < c.sparse_dim(); ++k) rows.push_back(c._indices().select(0, k));
      at::Tensor dv = at::index(d, rows);
      r = make_sparse(c, c.sparse_dim(), c.dense_dim(), c.sizes(), c._indices().clone(), (c._values() * dv).to(t), true);
    }
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
  if (mat1._nnz() > 0) {
    at::Tensor idx = mat1._indices();
    at::Tensor rows = mat2.to(t).index_select(0, idx.select(0, 1)) * mat1._values().to(t).unsqueeze(1);
    product.index_add_(0, idx.select(0, 0), rows);
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

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "mm: expected 2-D tensors");
  if (self.is_sparse() && mat2.is_sparse()) {  // sparse @ sparse: sparse
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
  at::Tensor m = mask.coalesce();
  at::Tensor src = self.is_sparse() ? self.to_dense() : self;
  c10::List<std::optional<at::Tensor>> rows;
  for (int64_t d = 0; d < m.sparse_dim(); ++d) rows.push_back(m._indices().select(0, d));
  at::Tensor values = m._nnz() == 0 || m.sparse_dim() == 0 ? at::empty({0}, src.options()) : at::index(src, rows);
  if (m.sparse_dim() == 0) values = src.unsqueeze(0).expand(m._values().sizes()).clone();
  if (m._nnz() == 0) {
    std::vector<int64_t> vshape(m._values().sizes().begin(), m._values().sizes().end());
    values = at::empty(vshape, src.options());
  }
  return make_sparse(m, m.sparse_dim(), m.dense_dim(), m.sizes(), m._indices().clone(), values, true);
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
  m.impl("_to_sparse", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, std::optional<at::Layout>, at::OptionalIntArrayRef, std::optional<int64_t>)>(sparse_coo_to_sparse)));
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
  if (crow.dim() != 1) {
    auto stack = on_cpu("aten::_convert_indices_from_csr_to_coo", "", crow, col, out_int32, transpose);
    return stack[0].toTensor();
  }
  at::Tensor c = crow.to(at::kLong);
  int64_t n = c.numel() - 1;
  at::Tensor counts = c.narrow(0, 1, n) - c.narrow(0, 0, n);
  at::Tensor rows = at::repeat_interleave(at::arange(n, c.options()), counts, 0, col.numel());
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
  return at::_sparse_compressed_tensor_unsafe(ccol, t._indices().select(0, 1).contiguous(), t._values(), self.sizes(),
                                              t._values().options().layout(at::kSparseCsc));
}

at::Tensor to_csc_op(const at::Tensor& self, std::optional<int64_t> dense_dim) { return to_csc(self, dense_dim); }

at::Tensor& csr_addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
                          const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  if (mat1.layout() == at::kSparseCsr && mat2.layout() == at::kStrided && mat1.dim() == 2 && mat1.dense_dim() == 0) {
    return addmm_out(self, csr_as_coo(mat1), mat2, beta, alpha, out);
  }
  if (mat1.layout() == at::kSparseCsr && mat2.layout() == at::kSparseCsr && out.layout() == at::kSparseCsr) {
    at::Tensor dense = at::addmm(self.layout() == at::kStrided ? self : self.to_dense(), mat1, mat2.to_dense(), beta, alpha);
    out.copy_(dense.to_sparse_csr());
    return out;
  }
  auto stack = on_cpu("aten::addmm", "out", self, mat1, mat2, beta, alpha, out);
  return out;
}

at::Tensor csr_addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                     const at::Scalar& alpha) {
  at::Tensor out = at::empty({0}, mat2.options().layout(at::kStrided).dtype(at::result_type(mat1, mat2)));
  return csr_addmm_out(self, mat1, mat2, beta, alpha, out);
}

at::Tensor csr_mm(const at::Tensor& self, const at::Tensor& mat2) {
  if (self.layout() == at::kSparseCsr && mat2.layout() == at::kStrided) {
    return vk::mm(csr_as_coo(self), mat2);
  }
  if (self.layout() == at::kStrided && mat2.layout() == at::kSparseCsr) {
    return vk::mm(self, csr_as_coo(mat2));
  }
  return vk::mm(self.layout() == at::kSparseCsr ? csr_as_coo(self) : self,
                mat2.layout() == at::kSparseCsr ? csr_as_coo(mat2) : mat2).to_sparse_csr();
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
  m.impl("_to_dense", TORCH_FN(sparse_compressed_to_dense));
  m.impl("_to_sparse.sparse_dim", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, int64_t)>(sparse_compressed_to_sparse)));
  m.impl("_to_sparse", TORCH_FN(static_cast<at::Tensor (*)(const at::Tensor&, std::optional<at::Layout>, at::OptionalIntArrayRef,
                                                            std::optional<int64_t>)>(sparse_compressed_to_sparse)));
  m.impl("sum", TORCH_FN(sum_csr));
  m.impl("sparse_mask", TORCH_FN(sparse_mask_sparse_compressed));
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
  m.impl("mm", TORCH_FN(vk::csr_mm));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_convert_indices_from_coo_to_csr", TORCH_FN(vk::convert_coo_to_csr));
  m.impl("_convert_indices_from_csr_to_coo", TORCH_FN(vk::convert_csr_to_coo));
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
at::Tensor compressed_via_dense(const at::Tensor& self, at::Layout layout, at::OptionalIntArrayRef blocksize,
                                std::optional<int64_t> dense_dim) {
  at::Tensor d = self.to_dense();
  int64_t dd = dense_dim.value_or(self.layout() == at::kStrided ? 0 : self.dense_dim());
  at::Tensor r = d.to_sparse(layout, blocksize, dd);
  at::ScalarType it = self.layout() == at::kStrided || self.layout() == at::kSparse ? at::kLong
                       : (self.layout() == at::kSparseCsr || self.layout() == at::kSparseBsr ? self.crow_indices()
                                                                                             : self.ccol_indices())
                             .scalar_type();
  bool rows = layout == at::kSparseCsr || layout == at::kSparseBsr;
  at::Tensor ci = rows ? r.crow_indices() : r.ccol_indices();
  at::Tensor pi = rows ? r.col_indices() : r.row_indices();
  // (indices contiguous per batch, as PyTorch's checks require)
  return at::_sparse_compressed_tensor_unsafe(ci.to(it).contiguous(), pi.to(it).contiguous(), r.values().contiguous(),
                                              r.sizes(), r.values().options().layout(layout));
}

at::Tensor to_csr_any(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  if (self.layout() == at::kSparseCsr && (!dense_dim.has_value() || *dense_dim == self.dense_dim())) return self.clone();
  return compressed_via_dense(self, at::kSparseCsr, std::nullopt, dense_dim);
}
at::Tensor to_csc_any(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  if (self.layout() == at::kSparseCsc && (!dense_dim.has_value() || *dense_dim == self.dense_dim())) return self.clone();
  if (self.dim() == 2 && (!dense_dim.has_value() || *dense_dim == 0) && self.layout() != at::kSparseBsr &&
      self.layout() != at::kSparseBsc && (self.layout() != at::kSparseCsr || self.dense_dim() == 0) &&
      (self.layout() != at::kSparse || self.dense_dim() == 0)) {
    at::Tensor r = to_csc(self, dense_dim);
    if (self.layout() == at::kSparseCsr && self.crow_indices().scalar_type() == at::kInt) {
      return at::_sparse_compressed_tensor_unsafe(r.ccol_indices().to(at::kInt), r.row_indices().to(at::kInt),
                                                  r.values(), r.sizes(), r.values().options().layout(at::kSparseCsc));
    }
    return r;
  }
  return compressed_via_dense(self, at::kSparseCsc, std::nullopt, dense_dim);
}
at::Tensor to_bsr_any(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
  return compressed_via_dense(self, at::kSparseBsr, blocksize, dense_dim);
}
at::Tensor to_bsc_any(const at::Tensor& self, at::IntArrayRef blocksize, std::optional<int64_t> dense_dim) {
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
  return vk::mm(a, b.to_dense()).to_sparse().coalesce();
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
  m.impl("conj_physical", TORCH_FN(conj_physical_sparse_csr));
  m.impl("conj_physical_", TORCH_FN(conj_physical_sparse_csr_));
  m.impl("conj_physical.out", TORCH_FN(conj_physical_sparse_csr_out));
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
  return at::_sparse_compressed_tensor_unsafe(ci.contiguous(), pi.contiguous(), r.values().contiguous(), r.sizes(),
                                              r.values().options().layout(r.layout()));
}

// dense or COO [*batch, rows, cols, *dense] -> CSR: unbatched through COO
// (its rows compressed by convert_coo_to_csr), batched by PyTorch's code.
at::Tensor to_csr_from(const at::Tensor& self, std::optional<int64_t> dense_dim) {
  int64_t dd = dense_dim.value_or(self.layout() == at::kSparse ? self.dense_dim() : 0);
  if (self.dim() - dd == 2) {
    at::Tensor coo = (self.layout() == at::kSparse ? self : self.to_sparse(2)).coalesce();
    if (coo.dense_dim() != dd) coo = coo.to_dense().to_sparse(2).coalesce();
    at::Tensor idx = coo._indices();
    at::Tensor crow = convert_coo_to_csr(idx.select(0, 0), self.size(0), false);
    return at::_sparse_compressed_tensor_unsafe(crow, idx.select(0, 1).contiguous(), coo._values().contiguous(),
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
  m.impl("_to_sparse_csc", TORCH_FN(vk::dense_to_csc));
  m.impl("_to_sparse_bsr", TORCH_FN(vk::dense_to_bsr));
  m.impl("_to_sparse_bsc", TORCH_FN(vk::dense_to_bsc));
  m.impl("_to_sparse_csr", TORCH_FN(vk::to_csr_from));
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl("_to_sparse_csr", TORCH_FN(vk::to_csr_from));
}
