//
// detail/winrt_timer_scheduler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINRT_TIMER_SCHEDULER_HPP
#define ASIO_DETAIL_WINRT_TIMER_SCHEDULER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include <cstddef>
#include "asio/detail/event.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/wait_op.hpp"
#include "asio/io_context.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/thread.hpp"
#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class winrt_timer_scheduler
  : public asio::detail::service_base<winrt_timer_scheduler>
{
public:
  // Constructor.
  ASIO_DECL winrt_timer_scheduler(asio::io_context& io_context);

  // Destructor.
  ASIO_DECL ~winrt_timer_scheduler();

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Recreate internal descriptors following a fork.
  ASIO_DECL void notify_fork(
      asio::io_context::fork_event fork_ev);

  // Initialise the task. No effect as this class uses its own thread.
  ASIO_DECL void init_task();

  // Add a new timer queue to the reactor.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& queue);

  // Remove a timer queue from the reactor.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& queue);

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& queue,
      const typename Time_Traits::time_type& time,
      typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op);

  // Cancel the timer operations associated with the given token. Returns the
  // number of operations that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& timer,
      std::size_t max_cancelled = (std::numeric_limits<std::size_t>::max)());

  // Move the timer operations associated with the given timer.
  template <typename Time_Traits>
  void move_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& to,
      typename timer_queue<Time_Traits>::per_timer_data& from);

private:
  // Run the select loop in the thread.
  ASIO_DECL void run_thread();

  // Entry point for the select loop thread.
  ASIO_DECL static void call_run_thread(winrt_timer_scheduler* reactor);

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // The io_context implementation used to post completions.
  io_context_impl& io_context_;

  // Mutex used to protect internal variables.
  asio::detail::mutex mutex_;

  // Event used to wake up background thread.
  asio::detail::event event_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // The background thread that is waiting for timers to expire.
  asio::detail::thread* thread_;

  // Does the background thread need to stop.
  bool stop_thread_;

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/winrt_timer_scheduler.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/winrt_timer_scheduler.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_TIMER_SCHEDULER_HPP
