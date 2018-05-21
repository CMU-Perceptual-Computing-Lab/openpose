//
// detail/impl/winrt_timer_scheduler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WINRT_TIMER_SCHEDULER_HPP
#define ASIO_DETAIL_IMPL_WINRT_TIMER_SCHEDULER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Time_Traits>
void winrt_timer_scheduler::add_timer_queue(timer_queue<Time_Traits>& queue)
{
  do_add_timer_queue(queue);
}

// Remove a timer queue from the reactor.
template <typename Time_Traits>
void winrt_timer_scheduler::remove_timer_queue(timer_queue<Time_Traits>& queue)
{
  do_remove_timer_queue(queue);
}

template <typename Time_Traits>
void winrt_timer_scheduler::schedule_timer(timer_queue<Time_Traits>& queue,
    const typename Time_Traits::time_type& time,
    typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op)
{
  asio::detail::mutex::scoped_lock lock(mutex_);

  if (shutdown_)
  {
    io_context_.post_immediate_completion(op, false);
    return;
  }

  bool earliest = queue.enqueue_timer(time, timer, op);
  io_context_.work_started();
  if (earliest)
    event_.signal(lock);
}

template <typename Time_Traits>
std::size_t winrt_timer_scheduler::cancel_timer(timer_queue<Time_Traits>& queue,
    typename timer_queue<Time_Traits>::per_timer_data& timer,
    std::size_t max_cancelled)
{
  asio::detail::mutex::scoped_lock lock(mutex_);
  op_queue<operation> ops;
  std::size_t n = queue.cancel_timer(timer, ops, max_cancelled);
  lock.unlock();
  io_context_.post_deferred_completions(ops);
  return n;
}

template <typename Time_Traits>
void winrt_timer_scheduler::move_timer(timer_queue<Time_Traits>& queue,
    typename timer_queue<Time_Traits>::per_timer_data& to,
    typename timer_queue<Time_Traits>::per_timer_data& from)
{
  asio::detail::mutex::scoped_lock lock(mutex_);
  op_queue<operation> ops;
  queue.cancel_timer(to, ops);
  queue.move_timer(to, from);
  lock.unlock();
  scheduler_.post_deferred_completions(ops);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_IMPL_WINRT_TIMER_SCHEDULER_HPP
