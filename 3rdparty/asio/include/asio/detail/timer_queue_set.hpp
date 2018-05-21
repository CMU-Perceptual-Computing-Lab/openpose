//
// detail/timer_queue_set.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TIMER_QUEUE_SET_HPP
#define ASIO_DETAIL_TIMER_QUEUE_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/timer_queue_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class timer_queue_set
{
public:
  // Constructor.
  ASIO_DECL timer_queue_set();

  // Add a timer queue to the set.
  ASIO_DECL void insert(timer_queue_base* q);

  // Remove a timer queue from the set.
  ASIO_DECL void erase(timer_queue_base* q);

  // Determine whether all queues are empty.
  ASIO_DECL bool all_empty() const;

  // Get the wait duration in milliseconds.
  ASIO_DECL long wait_duration_msec(long max_duration) const;

  // Get the wait duration in microseconds.
  ASIO_DECL long wait_duration_usec(long max_duration) const;

  // Dequeue all ready timers.
  ASIO_DECL void get_ready_timers(op_queue<operation>& ops);

  // Dequeue all timers.
  ASIO_DECL void get_all_timers(op_queue<operation>& ops);

private:
  timer_queue_base* first_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/timer_queue_set.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_TIMER_QUEUE_SET_HPP
