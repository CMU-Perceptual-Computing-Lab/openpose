//
// basic_signal_set.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SIGNAL_SET_HPP
#define ASIO_BASIC_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#include "asio/basic_io_object.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/signal_set_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Provides signal functionality.
/**
 * The basic_signal_set class template provides the ability to perform an
 * asynchronous wait for one or more signals to occur.
 *
 * Most applications will use the asio::signal_set typedef.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Example
 * Performing an asynchronous wait:
 * @code
 * void handler(
 *     const asio::error_code& error,
 *     int signal_number)
 * {
 *   if (!error)
 *   {
 *     // A signal occurred.
 *   }
 * }
 *
 * ...
 *
 * // Construct a signal set registered for process termination.
 * asio::signal_set signals(io_context, SIGINT, SIGTERM);
 *
 * // Start an asynchronous wait for one of the signals to occur.
 * signals.async_wait(handler);
 * @endcode
 *
 * @par Queueing of signal notifications
 *
 * If a signal is registered with a signal_set, and the signal occurs when
 * there are no waiting handlers, then the signal notification is queued. The
 * next async_wait operation on that signal_set will dequeue the notification.
 * If multiple notifications are queued, subsequent async_wait operations
 * dequeue them one at a time. Signal notifications are dequeued in order of
 * ascending signal number.
 *
 * If a signal number is removed from a signal_set (using the @c remove or @c
 * erase member functions) then any queued notifications for that signal are
 * discarded.
 *
 * @par Multiple registration of signals
 *
 * The same signal number may be registered with different signal_set objects.
 * When the signal occurs, one handler is called for each signal_set object.
 *
 * Note that multiple registration only works for signals that are registered
 * using Asio. The application must not also register a signal handler using
 * functions such as @c signal() or @c sigaction().
 *
 * @par Signal masking on POSIX platforms
 *
 * POSIX allows signals to be blocked using functions such as @c sigprocmask()
 * and @c pthread_sigmask(). For signals to be delivered, programs must ensure
 * that any signals registered using signal_set objects are unblocked in at
 * least one thread.
 */
template <typename SignalSetService = signal_set_service>
class basic_signal_set
  : public basic_io_object<SignalSetService>
{
public:
  /// Construct a signal set without adding any signals.
  /**
   * This constructor creates a signal set without registering for any signals.
   *
   * @param io_context The io_context object that the signal set will use to
   * dispatch handlers for any asynchronous operations performed on the set.
   */
  explicit basic_signal_set(asio::io_context& io_context)
    : basic_io_object<SignalSetService>(io_context)
  {
  }

  /// Construct a signal set and add one signal.
  /**
   * This constructor creates a signal set and registers for one signal.
   *
   * @param io_context The io_context object that the signal set will use to
   * dispatch handlers for any asynchronous operations performed on the set.
   *
   * @param signal_number_1 The signal number to be added.
   *
   * @note This constructor is equivalent to performing:
   * @code asio::signal_set signals(io_context);
   * signals.add(signal_number_1); @endcode
   */
  basic_signal_set(asio::io_context& io_context, int signal_number_1)
    : basic_io_object<SignalSetService>(io_context)
  {
    asio::error_code ec;
    this->get_service().add(this->get_implementation(), signal_number_1, ec);
    asio::detail::throw_error(ec, "add");
  }

  /// Construct a signal set and add two signals.
  /**
   * This constructor creates a signal set and registers for two signals.
   *
   * @param io_context The io_context object that the signal set will use to
   * dispatch handlers for any asynchronous operations performed on the set.
   *
   * @param signal_number_1 The first signal number to be added.
   *
   * @param signal_number_2 The second signal number to be added.
   *
   * @note This constructor is equivalent to performing:
   * @code asio::signal_set signals(io_context);
   * signals.add(signal_number_1);
   * signals.add(signal_number_2); @endcode
   */
  basic_signal_set(asio::io_context& io_context, int signal_number_1,
      int signal_number_2)
    : basic_io_object<SignalSetService>(io_context)
  {
    asio::error_code ec;
    this->get_service().add(this->get_implementation(), signal_number_1, ec);
    asio::detail::throw_error(ec, "add");
    this->get_service().add(this->get_implementation(), signal_number_2, ec);
    asio::detail::throw_error(ec, "add");
  }

  /// Construct a signal set and add three signals.
  /**
   * This constructor creates a signal set and registers for three signals.
   *
   * @param io_context The io_context object that the signal set will use to
   * dispatch handlers for any asynchronous operations performed on the set.
   *
   * @param signal_number_1 The first signal number to be added.
   *
   * @param signal_number_2 The second signal number to be added.
   *
   * @param signal_number_3 The third signal number to be added.
   *
   * @note This constructor is equivalent to performing:
   * @code asio::signal_set signals(io_context);
   * signals.add(signal_number_1);
   * signals.add(signal_number_2);
   * signals.add(signal_number_3); @endcode
   */
  basic_signal_set(asio::io_context& io_context, int signal_number_1,
      int signal_number_2, int signal_number_3)
    : basic_io_object<SignalSetService>(io_context)
  {
    asio::error_code ec;
    this->get_service().add(this->get_implementation(), signal_number_1, ec);
    asio::detail::throw_error(ec, "add");
    this->get_service().add(this->get_implementation(), signal_number_2, ec);
    asio::detail::throw_error(ec, "add");
    this->get_service().add(this->get_implementation(), signal_number_3, ec);
    asio::detail::throw_error(ec, "add");
  }

  /// Add a signal to a signal_set.
  /**
   * This function adds the specified signal to the set. It has no effect if the
   * signal is already in the set.
   *
   * @param signal_number The signal to be added to the set.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void add(int signal_number)
  {
    asio::error_code ec;
    this->get_service().add(this->get_implementation(), signal_number, ec);
    asio::detail::throw_error(ec, "add");
  }

  /// Add a signal to a signal_set.
  /**
   * This function adds the specified signal to the set. It has no effect if the
   * signal is already in the set.
   *
   * @param signal_number The signal to be added to the set.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID add(int signal_number, asio::error_code& ec)
  {
    this->get_service().add(this->get_implementation(), signal_number, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Remove a signal from a signal_set.
  /**
   * This function removes the specified signal from the set. It has no effect
   * if the signal is not in the set.
   *
   * @param signal_number The signal to be removed from the set.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Removes any notifications that have been queued for the specified
   * signal number.
   */
  void remove(int signal_number)
  {
    asio::error_code ec;
    this->get_service().remove(this->get_implementation(), signal_number, ec);
    asio::detail::throw_error(ec, "remove");
  }

  /// Remove a signal from a signal_set.
  /**
   * This function removes the specified signal from the set. It has no effect
   * if the signal is not in the set.
   *
   * @param signal_number The signal to be removed from the set.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Removes any notifications that have been queued for the specified
   * signal number.
   */
  ASIO_SYNC_OP_VOID remove(int signal_number,
      asio::error_code& ec)
  {
    this->get_service().remove(this->get_implementation(), signal_number, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Remove all signals from a signal_set.
  /**
   * This function removes all signals from the set. It has no effect if the set
   * is already empty.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Removes all queued notifications.
   */
  void clear()
  {
    asio::error_code ec;
    this->get_service().clear(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "clear");
  }

  /// Remove all signals from a signal_set.
  /**
   * This function removes all signals from the set. It has no effect if the set
   * is already empty.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Removes all queued notifications.
   */
  ASIO_SYNC_OP_VOID clear(asio::error_code& ec)
  {
    this->get_service().clear(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Cancel all operations associated with the signal set.
  /**
   * This function forces the completion of any pending asynchronous wait
   * operations against the signal set. The handler for each cancelled
   * operation will be invoked with the asio::error::operation_aborted
   * error code.
   *
   * Cancellation does not alter the set of registered signals.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note If a registered signal occurred before cancel() is called, then the
   * handlers for asynchronous wait operations will:
   *
   * @li have already been invoked; or
   *
   * @li have been queued for invocation in the near future.
   *
   * These handlers can no longer be cancelled, and therefore are passed an
   * error code that indicates the successful completion of the wait operation.
   */
  void cancel()
  {
    asio::error_code ec;
    this->get_service().cancel(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "cancel");
  }

  /// Cancel all operations associated with the signal set.
  /**
   * This function forces the completion of any pending asynchronous wait
   * operations against the signal set. The handler for each cancelled
   * operation will be invoked with the asio::error::operation_aborted
   * error code.
   *
   * Cancellation does not alter the set of registered signals.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note If a registered signal occurred before cancel() is called, then the
   * handlers for asynchronous wait operations will:
   *
   * @li have already been invoked; or
   *
   * @li have been queued for invocation in the near future.
   *
   * These handlers can no longer be cancelled, and therefore are passed an
   * error code that indicates the successful completion of the wait operation.
   */
  ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
  {
    this->get_service().cancel(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Start an asynchronous operation to wait for a signal to be delivered.
  /**
   * This function may be used to initiate an asynchronous wait against the
   * signal set. It always returns immediately.
   *
   * For each call to async_wait(), the supplied handler will be called exactly
   * once. The handler will be called when:
   *
   * @li One of the registered signals in the signal set occurs; or
   *
   * @li The signal set was cancelled, in which case the handler is passed the
   * error code asio::error::operation_aborted.
   *
   * @param handler The handler to be called when the signal occurs. Copies
   * will be made of the handler as required. The function signature of the
   * handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   int signal_number // Indicates which signal occurred.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   */
  template <typename SignalHandler>
  ASIO_INITFN_RESULT_TYPE(SignalHandler,
      void (asio::error_code, int))
  async_wait(ASIO_MOVE_ARG(SignalHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a SignalHandler.
    ASIO_SIGNAL_HANDLER_CHECK(SignalHandler, handler) type_check;

    return this->get_service().async_wait(this->get_implementation(),
        ASIO_MOVE_CAST(SignalHandler)(handler));
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_BASIC_SIGNAL_SET_HPP
