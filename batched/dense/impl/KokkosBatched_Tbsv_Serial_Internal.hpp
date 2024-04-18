//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSBATCHED_TBSV_SERIAL_INTERNAL_HPP_
#define KOKKOSBATCHED_TBSV_SERIAL_INTERNAL_HPP_

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

///
/// Serial Internal Impl
/// ====================

///
/// Lower
///

template <typename AlgoType>
struct SerialTbsvInternalLower {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const bool use_unit_diag,
                                           const bool do_conj, const int am,
                                           const int an, const int xm,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1,
                                           /**/ ValueType *KOKKOS_RESTRICT x,
                                           const int xs0, const int k,
                                           const int incx);
};

template <>
template <typename ValueType>
KOKKOS_INLINE_FUNCTION int
SerialTbsvInternalLower<Algo::Trsv::Unblocked>::invoke(
    const bool use_unit_diag, const bool do_conj, const int am, const int an,
    const int xn, const ValueType *KOKKOS_RESTRICT A, const int as0,
    const int as1,
    /**/ ValueType *KOKKOS_RESTRICT x, const int xs0, const int k,
    const int incx) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < an; ++j) {
    if (x[j * xs0] != 0) {
      if (do_conj) {
        if (!use_unit_diag) x[j * xs0] = x[j * xs0] / Kokkos::conj(A[0 + j * as1]);

        auto temp = x[j * xs0];
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i = j + 1; i < Kokkos::min(an, j + k + 1); ++i) {
          x[i * xs0] = x[i * xs0] - temp * Kokkos::conj(A[(i - j) * as0 + j * as1]);
        }
      }
    } else {
      if (!use_unit_diag) x[j * xs0] = x[j * xs0] / A[0 + j * as1];

      auto temp = x[j * xs0];
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i = j + 1; i < Kokkos::min(an, j + k + 1); ++i) {
        x[i * xs0] = x[i * xs0] - temp * A[(i - j) * as0 + j * as1];
      }        
    }
  }

  return 0;
}

///
/// Upper
///

template <typename AlgoType>
struct SerialTbsvInternalUpper {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const bool use_unit_diag,
                                           const bool do_conj, const int am,
                                           const int an, const int xm,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1,
                                           /**/ ValueType *KOKKOS_RESTRICT x,
                                           const int xs0, const int k,
                                           const int incx);
};

template <>
template <typename ValueType>
KOKKOS_INLINE_FUNCTION int
SerialTbsvInternalUpper<Algo::Trsv::Unblocked>::invoke(
    const bool use_unit_diag, const bool do_conj, const int am, const int an,
    const int xn, const ValueType *KOKKOS_RESTRICT A, const int as0,
    const int as1,
    /**/ ValueType *KOKKOS_RESTRICT x, const int xs0, const int k,
    const int incx) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = an-1; j >= 0; --j) {
    if (do_conj) {
      if (x[j * xs0] != 0) {
        if (!use_unit_diag) x[j * xs0] = x[j * xs0] / Kokkos::conj(A[k * as0 + j * as1]);

        auto temp = x[j * xs0];
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i = j - 1; i >= Kokkos::max(0, j - k); --i) {
          x[i * xs0] = x[i * xs0] - temp * Kokkos::conj(A[(k - j + i) * as0 + j * as1]);
        }
      }
    } else {
      if (x[j * xs0] != 0) {
        if (!use_unit_diag) x[j * xs0] = x[j * xs0] / A[k * as0 + j * as1];

        auto temp = x[j * xs0];
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i = j - 1; i >= Kokkos::max(0, j - k); --i) {
          x[i * xs0] = x[i * xs0] - temp * A[(k - j + i) * as0 + j * as1];
        }
      }
    }
  }

  return 0;
}

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_TBSV_SERIAL_INTERNAL_HPP_