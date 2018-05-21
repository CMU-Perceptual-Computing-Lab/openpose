//
// detail/kqueue_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2005 Stefan Arentz (stefan at soze dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_KQUEUE_REACTOR_HPP
#define ASIO_DETAIL_KQUEUE_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_KQUEUE)

#include <cstddef>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>
#include "asio/detail/limits.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/object_pool.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/wait_op.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"

// Older versions of Mac OS X may not define EV_OOBAND.
#if !defined(EV_OOBAND)
# define EV_OOBAND EV_FLAG1
#endif // !defined(EV_OOBAND)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class scheduler;

class kqueue_reactor
  : public execution_context_service_base<kqueue_reactor>
{
private:
  // The mutex type used by this reactor.
  typedef conditionally_enabled_mutex mutex;

public:
  enum op_types { read_op = 0, write_op = 1,
    connect_op = 1, except_op = 2, max_ops = 3 };

  // Per-descriptor queues.
  struct descriptor_state
  {
    descriptor_state(bool locking) : mutex_(locking) {}

    friend class kqueue_reactor;
    friend class object_pool_access;

    descriptor_state* next_;
    descriptor_state* prev_;

    mutex mutex_;
    int descriptor_;
    int num_kevents_; // 1 == read only, 2 == read and write
    op_queue<reactor_op> op_queue_[max_ops];
    bool shutdown_;
  };

  // Per-descriptor data.
  typedef descriptor_state* per_descriptor_data;

  // Constructor.
  ASIO_DECL kqueue_reactor(asio::execution_context& ctx);

  // Destructor.
  ASIO_DECL ~kqueue_reactor();

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

  // Run the kqueue loop.
  ASIO_DECL void run(long usec, op_queue<operation>& ops);

  // Interrupt the kqueue loop.
  ASIO_DECL void interrupt();

private:
  // Create the kqueue file descriptor. Throws an exception if the descriptor
  // cannot be created.
  ASIO_DECL static int do_kqueue_create();

  // Allocate a new descriptor state object.
  ASIO_DECL descriptor_state* allocate_descriptor_state();

  // Free an existing descriptor state object.
  ASIO_DECL void free_descriptor_state(descriptor_state* s);

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // Get the timeout value for the kevent call.
  ASIO_DECL timespec* get_timeout(long usec, timespec& ts);

  // The scheduler used to post completions.
  scheduler& scheduler_;

  // Mutex to protect access to internal data.
  mutex mutex_;

  // The kqueue file descriptor.
  int kqueue_fd_;

  // The interrupter is used to break a blocking kevent call.
  select_interrupter interrupter_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // Whether the service has been shut down.
  bool shutdown_;

  // Mutex to protect access to the registered descriptors.
  mutex registered_descriptors_mutex_;

  // Keep track of all registered descriptors.
  object_pool<descriptor_state> registered_descriptors_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/kqueue_reactor.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/kqueue_reactor.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_KQUEUE)

#endif // ASIO_DETAIL_KQUEUE_REACTOR_HPP
