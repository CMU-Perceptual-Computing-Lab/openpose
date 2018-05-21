//
// handler_invoke_hook.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_INVOKE_HOOK_HPP
#define ASIO_HANDLER_INVOKE_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/** @defgroup asio_handler_invoke asio::asio_handler_invoke
 *
 * @brief Default invoke function for handlers.
 *
 * Completion handlers for asynchronous operations are invoked by the
 * io_context associated with the corresponding object (e.g. a socket or
 * deadline_timer). Certain guarantees are made on when the handler may be
 * invoked, in particular that a handler can only be invoked from a thread that
 * is currently calling @c run() on the corresponding io_context object.
 * Handlers may subsequently be invoked through other objects (such as
 * io_context::strand objects) that provide additional guarantees.
 *
 * When asynchronous operations are composed from other asynchronous
 * operations, all intermediate handlers should be invoked using the same
 * method as the final handler. This is required to ensure that user-defined
 * objects are not accessed in a way that may violate the guarantees. This
 * hooking function ensures that the invoked method used for the final handler
 * is accessible at each intermediate step.
 *
 * Implement asio_handler_invoke for your own handlers to specify a custom
 * invocation strategy.
 *
 * This default implementation invokes the function object like so:
 * @code function(); @endcode
 * If necessary, the default implementation makes a copy of the function object
 * so that the non-const operator() can be used.
 *
 * @par Example
 * @code
 * class my_handler;
 *
 * template <typename Function>
 * void asio_handler_invoke(Function function, my_handler* context)
 * {
 *   context->strand_.dispatch(function);
 * }
 * @endcode
 */
/*@{*/

/// Default handler invocation hook used for non-const function objects.
template <typename Function>
inline void asio_handler_invoke(Function& function, ...)
{
  function();
}

/// Default handler invocation hook used for const function objects.
template <typename Function>
inline void asio_handler_invoke(const Function& function, ...)
{
  Function tmp(function);
  tmp();
}

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_HANDLER_INVOKE_HOOK_HPP
