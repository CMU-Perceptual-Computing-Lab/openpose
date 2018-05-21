//
// windows/object_handle_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2011 Boris Schaeling (boris@highscore.de)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_OBJECT_HANDLE_SERVICE_HPP
#define ASIO_WINDOWS_OBJECT_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

#include "asio/async_result.hpp"
#include "asio/detail/win_object_handle_service.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {

/// Default service implementation for an object handle.
class object_handle_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<object_handle_service>
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

private:
  // The type of the platform-specific implementation.
  typedef detail::win_object_handle_service service_impl_type;

public:
  /// The type of an object handle implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef service_impl_type::implementation_type implementation_type;
#endif

  /// The native handle type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef service_impl_type::native_handle_type native_handle_type;
#endif

  /// Construct a new object handle service for the specified io_context.
  explicit object_handle_service(asio::io_context& io_context)
    : asio::detail::service_base<object_handle_service>(io_context),
      service_impl_(io_context)
  {
  }

  /// Construct a new object handle implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a new object handle implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl)
  {
    service_impl_.move_construct(impl, other_impl);
  }

  /// Move-assign from another object handle implementation.
  void move_assign(implementation_type& impl,
      object_handle_service& other_service,
      implementation_type& other_impl)
  {
    service_impl_.move_assign(impl, other_service.service_impl_, other_impl);
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destroy an object handle implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Assign an existing native handle to an object handle.
  ASIO_SYNC_OP_VOID assign(implementation_type& impl,
      const native_handle_type& handle, asio::error_code& ec)
  {
    service_impl_.assign(impl, handle, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the handle is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Close an object handle implementation.
  ASIO_SYNC_OP_VOID close(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.close(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get the native handle implementation.
  native_handle_type native_handle(implementation_type& impl)
  {
    return service_impl_.native_handle(impl);
  }

  /// Cancel all asynchronous operations associated with the handle.
  ASIO_SYNC_OP_VOID cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.cancel(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  // Wait for a signaled state.
  void wait(implementation_type& impl, asio::error_code& ec)
  {
    service_impl_.wait(impl, ec);
  }

  /// Start an asynchronous wait.
  template <typename WaitHandler>
  ASIO_INITFN_RESULT_TYPE(WaitHandler,
      void (asio::error_code))
  async_wait(implementation_type& impl,
      ASIO_MOVE_ARG(WaitHandler) handler)
  {
    asio::async_completion<WaitHandler,
      void (asio::error_code)> init(handler);

    service_impl_.async_wait(impl, init.completion_handler);

    return init.result.get();
  }

private:
  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    service_impl_.shutdown();
  }

  // The platform-specific implementation.
  service_impl_type service_impl_;
};

} // namespace windows
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_WINDOWS_OBJECT_HANDLE_SERVICE_HPP
