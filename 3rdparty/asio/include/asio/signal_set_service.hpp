//
// signal_set_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SIGNAL_SET_SERVICE_HPP
#define ASIO_SIGNAL_SET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#include "asio/async_result.hpp"
#include "asio/detail/signal_set_service.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default service implementation for a signal set.
class signal_set_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<signal_set_service>
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

public:
  /// The type of a signal set implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef detail::signal_set_service::implementation_type implementation_type;
#endif

  /// Construct a new signal set service for the specified io_context.
  explicit signal_set_service(asio::io_context& io_context)
    : asio::detail::service_base<signal_set_service>(io_context),
      service_impl_(io_context)
  {
  }

  /// Construct a new signal set implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a signal set implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Add a signal to a signal_set.
  ASIO_SYNC_OP_VOID add(implementation_type& impl,
      int signal_number, asio::error_code& ec)
  {
    service_impl_.add(impl, signal_number, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Remove a signal to a signal_set.
  ASIO_SYNC_OP_VOID remove(implementation_type& impl,
      int signal_number, asio::error_code& ec)
  {
    service_impl_.remove(impl, signal_number, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Remove all signals from a signal_set.
  ASIO_SYNC_OP_VOID clear(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.clear(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Cancel all operations associated with the signal set.
  ASIO_SYNC_OP_VOID cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.cancel(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  // Start an asynchronous operation to wait for a signal to be delivered.
  template <typename SignalHandler>
  ASIO_INITFN_RESULT_TYPE(SignalHandler,
      void (asio::error_code, int))
  async_wait(implementation_type& impl,
      ASIO_MOVE_ARG(SignalHandler) handler)
  {
    async_completion<SignalHandler,
      void (asio::error_code, int)> init(handler);

    service_impl_.async_wait(impl, init.completion_handler);

    return init.result.get();
  }

private:
  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    service_impl_.shutdown();
  }

  // Perform any fork-related housekeeping.
  void notify_fork(asio::io_context::fork_event event)
  {
    service_impl_.notify_fork(event);
  }

  // The platform-specific implementation.
  detail::signal_set_service service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_SIGNAL_SET_SERVICE_HPP
