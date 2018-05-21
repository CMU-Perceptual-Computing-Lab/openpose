//
// detail/epoll_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_EPOLL_REACTOR_HPP
#define ASIO_DETAIL_EPOLL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_EPOLL)

#include "asio/detail/atomic_count.hpp"
#include "asio/detail/conditionally_enabled_mutex.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/object_pool.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/wait_op.hpp"
#include "asio/execution_context.hpp"

#if defined(ASIO_HAS_TIMERFD)
# include <sys/timerfd.h>
#endif // defined(ASIO_HAS_TIMERFD)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class epoll_reactor
  : public execution_context_service_base<epoll_reactor>
{
private:
  // The mutex type used by this reactor.
  typedef conditionally_enabled_mutex mutex;

public:
  enum op_types { read_op = 0, write_op = 1,
    connect_op = 1, except_op = 2, max_ops = 3 };

  // Per-descriptor queues.
  class descriptor_state : operation
  {
    friend class epoll_reactor;
    friend class object_pool_access;

    descriptor_state* next_;
    descriptor_state* prev_;

    mutex mutex_;
    epoll_reactor* reactor_;
    int descriptor_;
    uint32_t registered_events_;
    op_queue<reactor_op> op_queue_[max_ops];
    bool try_speculative_[max_ops];
    bool shutdown_;

    ASIO_DECL descriptor_state(bool locking);
    void set_ready_events(uint32_t events) { task_result_ = events; }
    void add_ready_events(uint32_t events) { task_result_ |= events; }
    ASIO_DECL operation* perform_io(uint32_t events);
    ASIO_DECL static void do_complete(
        void* owner, operation* base,
        const asio::error_code& ec, std::size_t bytes_transferred);
  };

  // Per-descriptor data.
  typedef descriptor_state* per_descriptor_data;

  // Constructor.
  ASIO_DECL epoll_reactor(asio::execution_context& ctx);

  // Destructor.
  ASIO_DECL ~epoll_reactor();

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Recreate internal descriptors following a fork.
  ASIO_DECL void notify_fork(
      asio::execution_context::fork_event fork_ev);

  // Initialise the task.
  ASIO_DECL void init_task();

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  ASIO_DECL int register_descriptor(socket_type descriptor,
      per_descriptor_data& descriptor_data);

  // Register a descriptor with an associated single operation. Returns 0 on
  // success, system error code on failure.
  ASIO_DECL int register_internal_descriptor(
      int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data, reactor_op* op);

  // Move descriptor registration from one descriptor_data object to another.
  ASIO_DECL void move_descriptor(socket_type descriptor,
      per_descriptor_data& target_descriptor_data,
      per_descriptor_data& source_descriptor_data);

  // Post a reactor operation for immediate completion.
  void post_immediate_completion(reactor_op* op, bool is_continuation)
  {
    scheduler_.post_immediate_completion(op, is_continuation);
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  ASIO_DECL void start_op(int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data, reactor_op* op,
      bool is_continuation, bool allow_speculative);

  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  ASIO_DECL void cancel_ops(socket_type descriptor,
      per_descriptor_data& descriptor_data);

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor. The reactor resources associated with
  // the descriptor must be released by calling cleanup_descriptor_data.
  ASIO_DECL void deregister_descriptor(socket_type descriptor,
      per_descriptor_data& descriptor_data, bool closing);

  // Remove the descriptor's registration from the reactor. The reactor
  // resources associated with the descriptor must be released by calling
  // cleanup_descriptor_data.
  ASIO_DECL void deregister_internal_descriptor(
      socket_type descriptor, per_descriptor_data& descriptor_data);

  // Perform any post-deregistration cleanup tasks associated with the
  // descriptor data.
  ASIO_DECL void cleanup_descriptor_data(
      per_descriptor_data& descriptor_data);

  // Add a new timer queue to the reactor.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& timer_queue);

  // Remove a timer queue from the reactor.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& timer_queue);

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

  // Run epoll once until interrupted or events are ready to be dispatched.
  ASIO_DECL void run(long usec, op_queue<operation>& ops);

  // Interrupt the select loop.
  ASIO_DECL void interrupt();

private:
  // The hint to pass to epoll_create to size its data structures.
  enum { epoll_size = 20000 };

  // Create the epoll file descriptor. Throws an exception if the descriptor
  // cannot be created.
  ASIO_DECL static int do_epoll_create();

  // Create the timerfd file descriptor. Does not throw.
  ASIO_DECL static int do_timerfd_create();

  // Allocate a new descriptor state object.
  ASIO_DECL descriptor_state* allocate_descriptor_state();

  // Free an existing descriptor state object.
  ASIO_DECL void free_descriptor_state(descriptor_state* s);

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // Called to recalculate and update the timeout.
  ASIO_DECL void update_timeout();

  // Get the timeout value for the epoll_wait call. The timeout value is
  // returned as a number of milliseconds. A return value of -1 indicates
  // that epoll_wait should block indefinitely.
  ASIO_DECL int get_timeout(int msec);

#if defined(ASIO_HAS_TIMERFD)
  // Get the timeout value for the timer descriptor. The return value is the
  // flag argument to be used when calling timerfd_settime.
  ASIO_DECL int get_timeout(itimerspec& ts);
#endif // defined(ASIO_HAS_TIMERFD)

  // The scheduler implementation used to post completions.
  scheduler& scheduler_;

  // Mutex to protect access to internal data.
  mutex mutex_;

  // The interrupter is used to break a blocking epoll_wait call.
  select_interrupter interrupter_;

  // The epoll file descriptor.
  int epoll_fd_;

  // The timer file descriptor.
  int timer_fd_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // Whether the service has been shut down.
  bool shutdown_;

  // Mutex to protect access to the registered descriptors.
  mutex registered_descriptors_mutex_;

  // Keep track of all registered descriptors.
  object_pool<descriptor_state> registered_descriptors_;

  // Helper class to do post-perform_io cleanup.
  struct perform_io_cleanup_on_block_exit;
  friend struct perform_io_cleanup_on_block_exit;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/epoll_reactor.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/epoll_reactor.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_EPOLL)

#endif // ASIO_DETAIL_EPOLL_REACTOR_HPP
