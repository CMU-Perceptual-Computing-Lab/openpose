//
// windows/object_handle.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2011 Boris Schaeling (boris@highscore.de)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_OBJECT_HANDLE_HPP
#define ASIO_WINDOWS_OBJECT_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

#include "asio/async_result.hpp"
#include "asio/basic_io_object.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/win_object_handle_service.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_ENABLE_OLD_SERVICES)
# include "asio/windows/basic_object_handle.hpp"
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#define ASIO_SVC_T asio::detail::win_object_handle_service

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {

#if defined(ASIO_ENABLE_OLD_SERVICES)
// Typedef for the typical usage of an object handle.
typedef basic_object_handle<> object_handle;
#else // defined(ASIO_ENABLE_OLD_SERVICES)
/// Provides object-oriented handle functionality.
/**
 * The windows::object_handle class provides asynchronous and blocking
 * object-oriented handle functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class object_handle
  : ASIO_SVC_ACCESS basic_io_object<ASIO_SVC_T>
{
public:
  /// The type of the executor associated with the object.
  typedef io_context::executor_type executor_type;

  /// The native representation of a handle.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef ASIO_SVC_T::native_handle_type native_handle_type;
#endif

  /// An object_handle is always the lowest layer.
  typedef object_handle lowest_layer_type;

  /// Construct an object_handle without opening it.
  /**
   * This constructor creates an object handle without opening it.
   *
   * @param io_context The io_context object that the object handle will use to
   * dispatch handlers for any asynchronous operations performed on the handle.
   */
  explicit object_handle(asio::io_context& io_context)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
  }

  /// Construct an object_handle on an existing native handle.
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
  object_handle(asio::io_context& io_context,
      const native_handle_type& native_handle)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
    asio::error_code ec;
    this->get_service().assign(this->get_implementation(), native_handle, ec);
    asio::detail::throw_error(ec, "assign");
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct an object_handle from another.
  /**
   * This constructor moves an object handle from one object to another.
   *
   * @param other The other object_handle object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c object_handle(io_context&) constructor.
   */
  object_handle(object_handle&& other)
    : basic_io_object<ASIO_SVC_T>(std::move(other))
  {
  }

  /// Move-assign an object_handle from another.
  /**
   * This assignment operator moves an object handle from one object to another.
   *
   * @param other The other object_handle object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c object_handle(io_context&) constructor.
   */
  object_handle& operator=(object_handle&& other)
  {
    basic_io_object<ASIO_SVC_T>::operator=(std::move(other));
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use get_executor().) Get the io_context associated with the
  /// object.
  /**
   * This function may be used to obtain the io_context object that the I/O
   * object uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the io_context object that the I/O object will use
   * to dispatch handlers. Ownership is not transferred to the caller.
   */
  asio::io_context& get_io_context()
  {
    return basic_io_object<ASIO_SVC_T>::get_io_context();
  }

  /// (Deprecated: Use get_executor().) Get the io_context associated with the
  /// object.
  /**
   * This function may be used to obtain the io_context object that the I/O
   * object uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the io_context object that the I/O object will use
   * to dispatch handlers. Ownership is not transferred to the caller.
   */
  asio::io_context& get_io_service()
  {
    return basic_io_object<ASIO_SVC_T>::get_io_service();
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Get the executor associated with the object.
  executor_type get_executor() ASIO_NOEXCEPT
  {
    return basic_io_object<ASIO_SVC_T>::get_executor();
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * layers. Since an object_handle cannot contain any further layers, it simply
   * returns a reference to itself.
   *
   * @return A reference to the lowest layer in the stack of layers. Ownership
   * is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Get a const reference to the lowest layer.
  /**
   * This function returns a const reference to the lowest layer in a stack of
   * layers. Since an object_handle cannot contain any further layers, it simply
   * returns a reference to itself.
   *
   * @return A const reference to the lowest layer in the stack of layers.
   * Ownership is not transferred to the caller.
   */
  const lowest_layer_type& lowest_layer() const
  {
    return *this;
  }

  /// Assign an existing native handle to the handle.
  /*
   * This function opens the handle to hold an existing native handle.
   *
   * @param handle A native handle.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void assign(const native_handle_type& handle)
  {
    asio::error_code ec;
    this->get_service().assign(this->get_implementation(), handle, ec);
    asio::detail::throw_error(ec, "assign");
  }

  /// Assign an existing native handle to the handle.
  /*
   * This function opens the handle to hold an existing native handle.
   *
   * @param handle A native handle.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID assign(const native_handle_type& handle,
      asio::error_code& ec)
  {
    this->get_service().assign(this->get_implementation(), handle, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the handle is open.
  bool is_open() const
  {
    return this->get_service().is_open(this->get_implementation());
  }

  /// Close the handle.
  /**
   * This function is used to close the handle. Any asynchronous read or write
   * operations will be cancelled immediately, and will complete with the
   * asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void close()
  {
    asio::error_code ec;
    this->get_service().close(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "close");
  }

  /// Close the handle.
  /**
   * This function is used to close the handle. Any asynchronous read or write
   * operations will be cancelled immediately, and will complete with the
   * asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID close(asio::error_code& ec)
  {
    this->get_service().close(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get the native handle representation.
  /**
   * This function may be used to obtain the underlying representation of the
   * handle. This is intended to allow access to native handle functionality
   * that is not otherwise provided.
   */
  native_handle_type native_handle()
  {
    return this->get_service().native_handle(this->get_implementation());
  }

  /// Cancel all asynchronous operations associated with the handle.
  /**
   * This function causes all outstanding asynchronous read or write operations
   * to finish immediately, and the handlers for cancelled operations will be
   * passed the asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void cancel()
  {
    asio::error_code ec;
    this->get_service().cancel(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "cancel");
  }

  /// Cancel all asynchronous operations associated with the handle.
  /**
   * This function causes all outstanding asynchronous read or write operations
   * to finish immediately, and the handlers for cancelled operations will be
   * passed the asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
  {
    this->get_service().cancel(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

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
    asio::async_completion<WaitHandler,
      void (asio::error_code)> init(handler);

    this->get_service().async_wait(this->get_implementation(),
        init.completion_handler);

    return init.result.get();
  }
};
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

} // namespace windows
} // namespace asio

#include "asio/detail/pop_options.hpp"

#undef ASIO_SVC_T

#endif // defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_WINDOWS_OBJECT_HANDLE_HPP
