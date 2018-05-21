//
// detail/win_iocp_io_context.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_IO_CONTEXT_HPP
#define ASIO_DETAIL_WIN_IOCP_IO_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/limits.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/thread_context.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/wait_op.hpp"
#include "asio/detail/win_iocp_operation.hpp"
#include "asio/detail/win_iocp_thread_info.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class wait_op;

class win_iocp_io_context
  : public execution_context_service_base<win_iocp_io_context>,
    public thread_context
{
public:
  // Constructor. Specifies a concurrency hint that is passed through to the
  // underlying I/O completion port.
  ASIO_DECL win_iocp_io_context(asio::execution_context& ctx,
      int concurrency_hint = -1);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Initialise the task. Nothing to do here.
  void init_task()
  {
  }

  // Register a handle with the IO completion port.
  ASIO_DECL asio::error_code register_handle(
      HANDLE handle, asio::error_code& ec);

  // Run the event loop until stopped or no more work.
  ASIO_DECL size_t run(asio::error_code& ec);

  // Run until stopped or one operation is performed.
  ASIO_DECL size_t run_one(asio::error_code& ec);

  // Run until timeout, interrupted, or one operation is performed.
  ASIO_DECL size_t wait_one(long usec, asio::error_code& ec);

  // Poll for operations without blocking.
  ASIO_DECL size_t poll(asio::error_code& ec);

  // Poll for one operation without blocking.
  ASIO_DECL size_t poll_one(asio::error_code& ec);

  // Stop the event processing loop.
  ASIO_DECL void stop();

  // Determine whether the io_context is stopped.
  bool stopped() const
  {
    return ::InterlockedExchangeAdd(&stopped_, 0) != 0;
  }

  // Restart in preparation for a subsequent run invocation.
  void restart()
  {
    ::InterlockedExchange(&stopped_, 0);
  }

  // Notify that some work has started.
  void work_started()
  {
    ::InterlockedIncrement(&outstanding_work_);
  }

  // Notify that some work has finished.
  void work_finished()
  {
    if (::InterlockedDecrement(&outstanding_work_) == 0)
      stop();
  }

  // Return whether a handler can be dispatched immediately.
  bool can_dispatch()
  {
    return thread_call_stack::contains(this) != 0;
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() has not yet been called for the operation.
  void post_immediate_completion(win_iocp_operation* op, bool)
  {
    work_started();
    post_deferred_completion(op);
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operation.
  ASIO_DECL void post_deferred_completion(win_iocp_operation* op);

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operations.
  ASIO_DECL void post_deferred_completions(
      op_queue<win_iocp_operation>& ops);

  // Request invocation of the given operation using the thread-private queue
  // and return immediately. Assumes that work_started() has not yet been
  // called for the operation.
  void post_private_immediate_completion(win_iocp_operation* op)
  {
    post_immediate_completion(op, false);
  }

  // Request invocation of the given operation using the thread-private queue
  // and return immediately. Assumes that work_started() was previously called
  // for the operation.
  void post_private_deferred_completion(win_iocp_operation* op)
  {
    post_deferred_completion(op);
  }

  // Enqueue the given operation following a failed attempt to dispatch the
  // operation for immediate invocation.
  void do_dispatch(operation* op)
  {
    post_immediate_completion(op, false);
  }

  // Process unfinished operations as part of a shutdown operation. Assumes
  // that work_started() was previously called for the operations.
  ASIO_DECL void abandon_operations(op_queue<operation>& ops);

  // Called after starting an overlapped I/O operation that did not complete
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  ASIO_DECL void on_pending(win_iocp_operation* op);

  // Called after starting an overlapped I/O operation that completed
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  ASIO_DECL void on_completion(win_iocp_operation* op,
      DWORD last_error = 0, DWORD bytes_transferred = 0);

  // Called after starting an overlapped I/O operation that completed
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  ASIO_DECL void on_completion(win_iocp_operation* op,
      const asio::error_code& ec, DWORD bytes_transferred = 0);

  // Add a new timer queue to the service.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& timer_queue);

  // Remove a timer queue from the service.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& timer_queue);

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& queue,
      const typename Time_Traits::time_type& time,
      typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op);

  // Cancel the timer associated with the given token. Returns the number of
  // handlers that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& timer,
      std::size_t max_cancelled = (std::numeric_limits<std::size_t>::max)());

  // Move the timer operations associated with the given timer.
  template <typename Time_Traits>
  void move_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& to,
      typename timer_queue<Time_Traits>::per_timer_data& from);

  // Get the concurrency hint that was used to initialise the io_context.
  int concurrency_hint() const
  {
    return concurrency_hint_;
  }

private:
#if defined(WINVER) && (WINVER < 0x0500)
  typedef DWORD dword_ptr_t;
  typedef ULONG ulong_ptr_t;
#else // defined(WINVER) && (WINVER < 0x0500)
  typedef DWORD_PTR dword_ptr_t;
  typedef ULONG_PTR ulong_ptr_t;
#endif // defined(WINVER) && (WINVER < 0x0500)

  // Dequeues at most one operation from the I/O completion port, and then
  // executes it. Returns the number of operations that were dequeued (i.e.
  // either 0 or 1).
  ASIO_DECL size_t do_one(DWORD msec, asio::error_code& ec);

  // Helper to calculate the GetQueuedCompletionStatus timeout.
  ASIO_DECL static DWORD get_gqcs_timeout();

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // Called to recalculate and update the timeout.
  ASIO_DECL void update_timeout();

  // Helper class to call work_finished() on block exit.
  struct work_finished_on_block_exit;

  // Helper class for managing a HANDLE.
  struct auto_handle
  {
    HANDLE handle;
    auto_handle() : handle(0) {}
    ~auto_handle() { if (handle) ::CloseHandle(handle); }
  };

  // The IO completion port used for queueing operations.
  auto_handle iocp_;

  // The count of unfinished work.
  long outstanding_work_;

  // Flag to indicate whether the event loop has been stopped.
  mutable long stopped_;

  // Flag to indicate whether there is an in-flight stop event. Every event
  // posted using PostQueuedCompletionStatus consumes non-paged pool, so to
  // avoid exhausting this resouce we limit the number of outstanding events.
  long stop_event_posted_;

  // Flag to indicate whether the service has been shut down.
  long shutdown_;

  enum
  {
    // Timeout to use with GetQueuedCompletionStatus on older versions of
    // Windows. Some versions of windows have a "bug" where a call to
    // GetQueuedCompletionStatus can appear stuck even though there are events
    // waiting on the queue. Using a timeout helps to work around the issue.
    default_gqcs_timeout = 500,

    // Maximum waitable timer timeout, in milliseconds.
    max_timeout_msec = 5 * 60 * 1000,

    // Maximum waitable timer timeout, in microseconds.
    max_timeout_usec = max_timeout_msec * 1000,

    // Completion key value used to wake up a thread to dispatch timers or
    // completed operations.
    wake_for_dispatch = 1,

    // Completion key value to indicate that an operation has posted with the
    // original last_error and bytes_transferred values stored in the fields of
    // the OVERLAPPED structure.
    overlapped_contains_result = 2
  };

  // Timeout to use with GetQueuedCompletionStatus.
  const DWORD gqcs_timeout_;

  // Function object for processing timeouts in a background thread.
  struct timer_thread_function;
  friend struct timer_thread_function;

  // Background thread used for processing timeouts.
  scoped_ptr<thread> timer_thread_;

  // A waitable timer object used for waiting for timeouts.
  auto_handle waitable_timer_;

  // Non-zero if timers or completed operations need to be dispatched.
  long dispatch_required_;

  // Mutex for protecting access to the timer queues and completed operations.
  mutex dispatch_mutex_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // The operations that are ready to dispatch.
  op_queue<win_iocp_operation> completed_ops_;

  // The concurrency hint used to initialise the io_context.
  const int concurrency_hint_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/win_iocp_io_context.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_iocp_io_context.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_IO_CONTEXT_HPP
