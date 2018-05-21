//
// windows/basic_object_handle.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2011 Boris Schaeling (boris@highscore.de)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_BASIC_OBJECT_HANDLE_HPP
#define ASIO_WINDOWS_BASIC_OBJECT_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/windows/basic_handle.hpp"
#include "asio/windows/object_handle_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {

/// Provides object-oriented handle functionality.
/**
 * The windows::basic_object_handle class template provides asynchronous and
 * blocking object-oriented handle functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <typename ObjectHandleService = object_handle_service>
class basic_object_handle
  : public basic_handle<ObjectHandleService>
{
public:
  /// The native representation of a handle.
  typedef typename ObjectHandleService::native_handle_type native_handle_type;

  /// Construct a basic_object_handle without opening it.
  /**
   * This constructor creates an object handle without opening it.
   *
   * @param io_context The io_context object that the object handle will use to
   * dispatch handlers for any asynchronous operations performed on the handle.
   */
  explicit basic_object_handle(asio::io_context& io_context)
    : basic_handle<ObjectHandleService>(io_context)
  {
  }

  /// Construct a basic_object_handle on an existing native handle.
  /**
   * This constructor creates an object handle object to hold an existing native
   * handle.
   *
   * @param io_context The io_context object that the object handle will use to
   * dispatch handlers for any asynchronous operations performed on the handle.
   *
   * @param native_handle The new underlying handle implementation.
   *
   * @throws asio::system_error Thrown on failure.
   */
  basic_object_handle(asio::io_context& io_context,
      const native_handle_type& native_handle)
    : basic_handle<ObjectHandleService>(io_context, native_handle)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a basic_object_handle from another.
  /**
   * This constructor moves an object handle from one object to another.
   *
   * @param other The other basic_object_handle object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c basic_object_handle(io_context&) constructor.
   */
  basic_object_handle(basic_object_handle&& other)
    : basic_handle<ObjectHandleService>(
        ASIO_MOVE_CAST(basic_object_handle)(other))
  {
  }

  /// Move-assign a basic_object_handle from another.
  /**
   * This assignment operator moves an object handle from one object to another.
   *
   * @param other The other basic_object_handle object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c basic_object_handle(io_context&) constructor.
   */
  basic_object_handle& operator=(basic_object_handle&& other)
  {
    basic_handle<ObjectHandleService>::operator=(
        ASIO_MOVE_CAST(basic_object_handle)(other));
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Perform a blocking wait on the object handle.
  /**
   * This function is used to wait for the object handle to be set to the
   * signalled state. This function blocks and does not return until the object
   * handle has been set to the signalled state.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void wait()
  {
    asio::error_code ec;
    this->get_service().wait(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "wait");
  }

  /// Perform a blocking wait on the object handle.
  /**
   * This function is used to wait for the object handle to be set to the
   * signalled state. This function blocks and does not return until the object
   * handle has been set to the signalled state.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  void wait(asio::error_code& ec)
  {
    this->get_service().wait(this->get_implementation(), ec);
  }

  /// Start an asynchronous wait on the object handle.
  /**
   * This function is be used to initiate an asynchronous wait against the
   * object handle. It always returns immediately.
   *
   * @param handler The handler to be called when the object handle is set to
   * the signalled state. Copies will be made of the handler as required. The
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error // Result of operation.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   */
  template <typename WaitHandler>
  ASIO_INITFN_RESULT_TYPE(WaitHandler,
      void (asio::error_code))
  async_wait(ASIO_MOVE_ARG(WaitHandler) handler)
  {
    return this->get_service().async_wait(this->get_implementation(),
        ASIO_MOVE_CAST(WaitHandler)(handler));
  }
};

} // namespace windows
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_WINDOWS_BASIC_OBJECT_HANDLE_HPP
