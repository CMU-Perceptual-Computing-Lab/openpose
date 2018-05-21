//
// thread.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_THREAD_HPP
#define ASIO_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/thread.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// A simple abstraction for starting threads.
/**
 * The asio::thread class implements the smallest possible subset of the
 * functionality of boost::thread. It is intended to be used only for starting
 * a thread and waiting for it to exit. If more extensive threading
 * capabilities are required, you are strongly advised to use something else.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Example
 * A typical use of asio::thread would be to launch a thread to run an
 * io_context's event processing loop:
 *
 * @par
 * @code asio::io_context io_context;
 * // ...
 * asio::thread t(boost::bind(&asio::io_context::run, &io_context));
 * // ...
 * t.join(); @endcode
 */
class thread
  : private noncopyable
{
public:
  /// Start a new thread that executes the supplied function.
  /**
   * This constructor creates a new thread that will execute the given function
   * or function object.
   *
   * @param f The function or function object to be run in the thread. The
   * function signature must be: @code void f(); @endcode
   */
  template <typename Function>
  explicit thread(Function f)
    : impl_(f)
  {
  }

  /// Destructor.
  ~thread()
  {
  }

  /// Wait for the thread to exit.
  /**
   * This function will block until the thread has exited.
   *
   * If this function is not called before the thread object is destroyed, the
   * thread itself will continue to run until completion. You will, however,
   * no longer have the ability to wait for it to exit.
   */
  void join()
  {
    impl_.join();
  }

private:
  detail::thread impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_THREAD_HPP
