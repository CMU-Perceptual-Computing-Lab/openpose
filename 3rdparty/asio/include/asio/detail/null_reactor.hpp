//
// detail/null_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_REACTOR_HPP
#define ASIO_DETAIL_NULL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/scheduler_operation.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_reactor
  : public execution_context_service_base<null_reactor>
{
public:
  // Constructor.
  null_reactor(asio::execution_context& ctx)
    : execution_context_service_base<null_reactor>(ctx)
  {
  }

  // Destructor.
  ~null_reactor()
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
  }

  // No-op because should never be called.
  void run(long /*usec*/, op_queue<scheduler_operation>& /*ops*/)
  {
  }

  // No-op.
  void interrupt()
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_NULL_REACTOR_HPP
