//
// detail/select_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SELECT_REACTOR_HPP
#define ASIO_DETAIL_SELECT_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP) \
  || (!defined(ASIO_HAS_DEV_POLL) \
      && !defined(ASIO_HAS_EPOLL) \
      && !defined(ASIO_HAS_KQUEUE) \
      && !defined(ASIO_WINDOWS_RUNTIME))

#include <cstddef>
#include "asio/detail/fd_set_adapter.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/wait_op.hpp"
#include "asio/execution_context.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/thread.hpp"
#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class select_reactor
  : public execution_context_service_base<select_reactor>
{
public:
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  enum op_types { read_op = 0, write_op = 1, except_op = 2,
    max_select_ops = 3, connect_op = 3, max_ops = 4 };
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  enum op_types { read_op = 0, write_op = 1, except_op = 2,
    max_select_ops = 3, connect_op = 1, max_ops = 3 };
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

  // Per-descriptor data.
  struct per_descriptor_data
  {
  };

  // Constructor.
  ASIO_DECL select_reactor(asio::execution_context& ctx);

  // Destructor.
  ASIO_DECL ~select_reactor();

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Recreate internal descriptors following a fork.
  ASIO_DECL void notify_fork(
      asio::execution_context::fork_event fork_ev);

  // Initialise the task, but only if the reactor is not in its own thread.
  ASIO_DECL void init_task();

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  ASIO_DECL int register_descriptor(socket_type, per_descriptor_data&);

  // Register a descriptor with an associated single operation. Returns 0 on
  // success, system error code on failure.
  ASIO_DECL int register_internal_descriptor(
      int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data, reactor_op* op);

  // Post a reactor operation for immediate completion.
  void post_immediate_completion(reactor_op* op, bool is_continuation)
  {
    scheduler_.post_immediate_completion(op, is_continuation);
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  ASIO_DECL void start_op(int op_type, socket_type descriptor,
      per_descriptor_data&, reactor_op* op, bool is_continuation, bool);

  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  ASIO_DECL void cancel_ops(socket_type descriptor, per_descriptor_data&);

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor. The reactor resources associated with
  // the descriptor must be released by calling cleanup_descriptor_data.
  ASIO_DECL void deregister_descriptor(socket_type descriptor,
      per_descriptor_data&, bool closing);

  // Remove the descriptor's registration from the reactor. The reactor
  // resources associated with the descriptor must be released by calling
  // cleanup_descriptor_data.
  ASIO_DECL void deregister_internal_descriptor(
      socket_type descriptor, per_descriptor_data&);

  // Perform any post-deregistration cleanup tasks associated with the
  // descriptor data.
  ASIO_DECL void cleanup_descriptor_data(per_descriptor_data&);

  // Move descriptor registration from one descriptor_data object to another.
  ASIO_DECL void move_descriptor(socket_type descriptor,
      per_descriptor_data& target_descriptor_data,
      per_descriptor_data& source_descriptor_data);

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
      typename timer_queue<Time_Traits>::per_timer_data& target,
      typename timer_queue<Time_Traits>::per_timer_data& source);

  // Run select once until interrupted or events are ready to be dispatched.
  ASIO_DECL void run(long usec, op_queue<operation>& ops);

  // Interrupt the select loop.
  ASIO_DECL void interrupt();

private:
#if defined(ASIO_HAS_IOCP)
  // Run the select loop in the thread.
  ASIO_DECL void run_thread();
#endif // defined(ASIO_HAS_IOCP)

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // Get the timeout value for the select call.
  ASIO_DECL timeval* get_timeout(long usec, timeval& tv);

  // Cancel all operations associated with the given descriptor. This function
  // does not acquire the select_reactor's mutex.
  ASIO_DECL void cancel_ops_unlocked(socket_type descriptor,
      const asio::error_code& ec);

  // The scheduler implementation used to post completions.
# if defined(ASIO_HAS_IOCP)
  typedef class win_iocp_io_context scheduler_type;
# else // defined(ASIO_HAS_IOCP)
  typedef class scheduler scheduler_type;
# endif // defined(ASIO_HAS_IOCP)
  scheduler_type& scheduler_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queues of read, write and except operations.
  reactor_op_queue<socket_type> op_queue_[max_ops];

  // The file descriptor sets to be passed to the select system call.
  fd_set_adapter fd_sets_[max_select_ops];

  // The timer queues.
  timer_queue_set timer_queues_;

#if defined(ASIO_HAS_IOCP)
  // Helper class to run the reactor loop in a thread.
  class thread_function;
  friend class thread_function;

  // Does the reactor loop thread need to stop.
  bool stop_thread_;

  // The thread that is running the reactor loop.
  asio::detail::thread* thread_;
#endif // defined(ASIO_HAS_IOCP)

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/select_reactor.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/select_reactor.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_IOCP)
       //   || (!defined(ASIO_HAS_DEV_POLL)
       //       && !defined(ASIO_HAS_EPOLL)
       //       && !defined(ASIO_HAS_KQUEUE)
       //       && !defined(ASIO_WINDOWS_RUNTIME))

#endif // ASIO_DETAIL_SELECT_REACTOR_HPP
