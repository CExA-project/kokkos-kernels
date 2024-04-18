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
#ifndef KOKKOSBATCHED_TBSV_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_TBSV_SERIAL_IMPL_HPP_

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Tbsv_Serial_Internal.hpp"

namespace KokkosBatched {
//// Lower non-transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Lower, Trans::NoTranspose, ArgDiag,
                  Algo::Tbsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalLower<Algo::Tbsv::Unblocked>::invoke(
        ArgDiag::use_unit_diag, false, A.extent(0), A.extent(1), x.extent(0),
        A.data(), A.stride_0(), A.stride_1(), x.data(), x.stride_0(), k, incx);
  }
};

//// Lower transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Lower, Trans::Transpose, ArgDiag,
                  Algo::Tbsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalLower<Algo::Tbsv::Unblocked>::invoke(
        ArgDiag::use_unit_diag, false, A.extent(1), A.extent(0), x.extent(0),
        A.data(), A.stride_1(), A.stride_0(), x.data(), x.stride_0(), k, incx);
  }
};

//// Lower conjugate-transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Lower, Trans::ConjTranspose, ArgDiag,
                  Algo::Tbsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalLower<Algo::Tbsv::Unblocked>::invoke(
        ArgDiag::use_unit_diag, true, A.extent(1), A.extent(0), x.extent(0),
        A.data(), A.stride_1(), A.stride_0(), x.data(), x.stride_0(), k, incx);
  }
};

//// Upper non-transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Upper, Trans::NoTranspose, ArgDiag,
                  Algo::Tbsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalUpper<Algo::Tbsv::Unblocked>::invoke(
        ArgDiag::use_unit_diag, false, A.extent(0), A.extent(1), x.extent(0),
        A.data(), A.stride_0(), A.stride_1(), x.data(), x.stride_0(), k, incx);
  }
};

//// Upper transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Upper, Trans::Transpose, ArgDiag,
                  Algo::Trsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalUpper<Algo::Trmm::Unblocked>::invoke(
        ArgDiag::use_unit_diag, false, A.extent(1), A.extent(0), x.extent(0),
        A.data(), A.stride_1(), A.stride_0(), x.data(), x.stride_0(), k, incx);
  }
};

//// Upper conjugate-transpose ////
template <typename ArgDiag>
struct SerialTbsv<Uplo::Upper, Trans::ConjTranspose, ArgDiag,
                  Algo::Trsv::Unblocked> {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &x, const int k,
                                           const int incx) {
    return SerialTbsvInternalUpper<Algo::Trmm::Unblocked>::invoke(
        ArgDiag::use_unit_diag, true, A.extent(1), A.extent(0), x.extent(0),
        A.data(), A.stride_1(), A.stride_0(), x.data(), x.stride_0(), k, incx);
  }
};

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_TBSV_SERIAL_IMPL_HPP_