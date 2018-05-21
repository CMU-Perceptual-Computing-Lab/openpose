//
// detail/signal_op.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SIGNAL_OP_HPP
#define ASIO_DETAIL_SIGNAL_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class signal_op
  : public operation
{
public:
  // The error code to be passed to the completion handler.
  asio::error_code ec_;

  // The signal number to be passed to the completion handler.
  int signal_number_;

protected:
  signal_op(func_type func)
    : operation(func),
      signal_number_(0)
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SIGNAL_OP_HPP
