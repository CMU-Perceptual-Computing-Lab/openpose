//
// detail/gcc_hppa_fenced_block.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_GCC_HPPA_FENCED_BLOCK_HPP
#define ASIO_DETAIL_GCC_HPPA_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(__GNUC__) && (defined(__hppa) || defined(__hppa__))

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class gcc_hppa_fenced_block
  : private noncopyable
{
public:
  enum half_t { half };
  enum full_t { full };

  // Constructor for a half fenced block.
  explicit gcc_hppa_fenced_block(half_t)
  {
  }

  // Constructor for a full fenced block.
  explicit gcc_hppa_fenced_block(full_t)
  {
    barrier();
  }

  // Destructor.
  ~gcc_hppa_fenced_block()
  {
    barrier();
  }

private:
  static void barrier()
  {
    // This is just a placeholder and almost certainly not sufficient.
    __asm__ __volatile__ ("" : : : "memory");
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(__GNUC__) && (defined(__hppa) || defined(__hppa__))

#endif // ASIO_DETAIL_GCC_HPPA_FENCED_BLOCK_HPP
