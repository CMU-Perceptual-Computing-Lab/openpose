//
// detail/wait_op.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WAIT_OP_HPP
#define ASIO_DETAIL_WAIT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class wait_op
  : public operation
{
public:
  // The error code to be passed to the completion handler.
  asio::error_code ec_;

protected:
  wait_op(func_type func)
    : operation(func)
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WAIT_OP_HPP
