//
// detail/strand_executor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STRAND_EXECUTOR_SERVICE_HPP
#define ASIO_DETAIL_STRAND_EXECUTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/atomic_count.hpp"
#include "asio/detail/executor_op.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/scheduler_operation.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Default service implementation for a strand.
class strand_executor_service
  : public execution_context_service_base<strand_executor_service>
{
public:
  // The underlying implementation of a strand.
  class strand_impl
  {
  public:
    ASIO_DECL ~strand_impl();

  private:
    friend class strand_executor_service;

    // Mutex to protect access to internal data.
    mutex* mutex_;

    // Indicates whether the strand is currently "locked" by a handler. This
    // means that there is a handler upcall in progress, or that the strand
    // itself has been scheduled in order to invoke some pending handlers.
    bool locked_;

    // Indicates that the strand has been shut down and will accept no further
    // handlers.
    bool shutdown_;

    // The handlers that are waiting on the strand but should not be run until
    // after the next time the strand is scheduled. This queue must only be
    // modified while the mutex is locked.
    op_queue<scheduler_operation> waiting_queue_;

    // The handlers that are ready to be run. Logically speaking, these are the
    // handlers that hold the strand's lock. The ready queue is only modified
    // from within the strand and so may be accessed without locking the mutex.
    op_queue<scheduler_operation> ready_queue_;

    // Pointers to adjacent handle implementations in linked list.
    strand_impl* next_;
    strand_impl* prev_;

    // The strand service in where the implementation is held.
    strand_executor_service* service_;
  };

  typedef shared_ptr<strand_impl> implementation_type;

  // Construct a new strand service for the specified context.
  ASIO_DECL explicit strand_executor_service(execution_context& context);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Create a new strand_executor implementation.
  ASIO_DECL implementation_type create_implementation();

  // Request invocation of the given function.
  template <typename Executor, typename Function, typename Allocator>
  static void dispatch(const implementation_type& impl, Executor& ex,
      ASIO_MOVE_ARG(Function) function, const Allocator& a);

  // Request invocation of the given function and return immediately.
  template <typename Executor, typename Function, typename Allocator>
  static void post(const implementation_type& impl, Executor& ex,
      ASIO_MOVE_ARG(Function) function, const Allocator& a);

  // Request invocation of the given function and return immediately.
  template <typename Executor, typename Function, typename Allocator>
  static void defer(const implementation_type& impl, Executor& ex,
      ASIO_MOVE_ARG(Function) function, const Allocator& a);

  // Determine whether the strand is running in the current thread.
  ASIO_DECL static bool running_in_this_thread(
      const implementation_type& impl);

private:
  friend class strand_impl;
  template <typename Executor> class invoker;

  // Adds a function to the strand. Returns true if it acquires the lock.
  ASIO_DECL static bool enqueue(const implementation_type& impl,
      scheduler_operation* op);

  // Mutex to protect access to the service-wide state.
  mutex mutex_;

  // Number of mutexes shared between all strand objects.
  enum { num_mutexes = 193 };

  // Pool of mutexes.
  scoped_ptr<mutex> mutexes_[num_mutexes];

  // Extra value used when hashing to prevent recycled memory locations from
  // getting the same mutex.
  std::size_t salt_;

  // The head of a linked list of all implementations.
  strand_impl* impl_list_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/strand_executor_service.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/strand_executor_service.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_STRAND_EXECUTOR_SERVICE_HPP
