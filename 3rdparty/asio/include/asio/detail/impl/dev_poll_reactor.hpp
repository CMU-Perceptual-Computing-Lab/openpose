//
// detail/impl/dev_poll_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_DEV_POLL_REACTOR_HPP
#define ASIO_DETAIL_IMPL_DEV_POLL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_DEV_POLL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Time_Traits>
void dev_poll_reactor::add_timer_queue(timer_queue<Time_Traits>& queue)
{
  do_add_timer_queue(queue);
}

template <typename Time_Traits>
void dev_poll_reactor::remove_timer_queue(timer_queue<Time_Traits>& queue)
{
  do_remove_timer_queue(queue);
}

template <typename Time_Traits>
void dev_poll_reactor::schedule_timer(timer_queue<Time_Traits>& queue,
    const typename Time_Traits::time_type& time,
    typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op)
{
  asio::detail::mutex::scoped_lock lock(mutex_);

  if (shutdown_)
  {
    scheduler_.post_immediate_completion(op, false);
    return;
  }

  bool earliest = queue.enqueue_timer(time, timer, op);
  scheduler_.work_started();
  if (earliest)
    interrupter_.interrupt();
}

template <typename Time_Traits>
std::size_t dev_poll_reactor::cancel_timer(timer_queue<Time_Traits>& queue,
    typename timer_queue<Time_Traits>::per_timer_data& timer,
    std::size_t max_cancelled)
{
  asio::detail::mutex::scoped_lock lock(mutex_);
  op_queue<operation> ops;
  std::size_t n = queue.cancel_timer(timer, ops, max_cancelled);
  lock.unlock();
  scheduler_.post_deferred_completions(ops);
  return n;
}

template <typename Time_Traits>
void dev_poll_reactor::move_timer(timer_queue<Time_Traits>& queue,
    typename timer_queue<Time_Traits>::per_timer_data& target,
    typename timer_queue<Time_Traits>::per_timer_data& source)
{
  asio::detail::mutex::scoped_lock lock(mutex_);
  op_queue<operation> ops;
  queue.cancel_timer(target, ops);
  queue.move_timer(target, source);
  lock.unlock();
  scheduler_.post_deferred_completions(ops);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_DEV_POLL)

#endif // ASIO_DETAIL_IMPL_DEV_POLL_REACTOR_HPP
