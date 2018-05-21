//
// detail/null_fenced_block.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_FENCED_BLOCK_HPP
#define ASIO_DETAIL_NULL_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_fenced_block
  : private noncopyable
{
public:
  enum half_or_full_t { half, full };

  // Constructor.
  explicit null_fenced_block(half_or_full_t)
  {
  }

  // Destructor.
  ~null_fenced_block()
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_NULL_FENCED_BLOCK_HPP
