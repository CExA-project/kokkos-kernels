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
#ifndef KOKKOSBATCHED_TBSV_DECL_HPP_
#define KOKKOSBATCHED_TBSV_DECL_HPP_

#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Vector.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

///
/// Serial Tbsv
///

template <typename ArgUplo, typename ArgTrans, typename ArgDiag,
          typename ArgAlgo>
struct SerialTbsv {
  template <typename AViewType, typename XViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const XViewType &X, const int k,
                                           const int incx);
};
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_TBSV_DECL_HPP_