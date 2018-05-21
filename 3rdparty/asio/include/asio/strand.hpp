//
// strand.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STRAND_HPP
#define ASIO_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/strand_executor_service.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Provides serialised function invocation for any executor type.
template <typename Executor>
class strand
{
public:
  /// The type of the underlying executor.
  typedef Executor inner_executor_type;

  /// Default constructor.
  /**
   * This constructor is only valid if the underlying executor type is default
   * constructible.
   */
  strand()
    : executor_(),
      impl_(use_service<detail::strand_executor_service>(
            executor_.context()).create_implementation())
  {
  }

  /// Construct a strand for the specified executor.
  explicit strand(const Executor& e)
    : executor_(e),
      impl_(use_service<detail::strand_executor_service>(
            executor_.context()).create_implementation())
  {
  }

  /// Copy constructor.
  strand(const strand& other) ASIO_NOEXCEPT
    : executor_(other.executor_),
      impl_(other.impl_)
  {
  }

  /// Converting constructor.
  /**
   * This constructor is only valid if the @c OtherExecutor type is convertible
   * to @c Executor.
   */
  template <class OtherExecutor>
  strand(
      const strand<OtherExecutor>& other) ASIO_NOEXCEPT
    : executor_(other.executor_),
      impl_(other.impl_)
  {
  }

  /// Assignment operator.
  strand& operator=(const strand& other) ASIO_NOEXCEPT
  {
    executor_ = other.executor_;
    impl_ = other.impl_;
    return *this;
  }

  /// Converting assignment operator.
  /**
   * This assignment operator is only valid if the @c OtherExecutor type is
   * convertible to @c Executor.
   */
  template <class OtherExecutor>
  strand& operator=(
      const strand<OtherExecutor>& other) ASIO_NOEXCEPT
  {
    executor_ = other.executor_;
    impl_ = other.impl_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move constructor.
  strand(strand&& other) ASIO_NOEXCEPT
    : executor_(ASIO_MOVE_CAST(Executor)(other.executor_)),
      impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
  {
  }

  /// Converting move constructor.
  /**
   * This constructor is only valid if the @c OtherExecutor type is convertible
   * to @c Executor.
   */
  template <class OtherExecutor>
  strand(strand<OtherExecutor>&& other) ASIO_NOEXCEPT
    : executor_(ASIO_MOVE_CAST(OtherExecutor)(other)),
      impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
  {
  }

  /// Move assignment operator.
  strand& operator=(strand&& other) ASIO_NOEXCEPT
  {
    executor_ = ASIO_MOVE_CAST(Executor)(other);
    impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
    return *this;
  }

  /// Converting move assignment operator.
  /**
   * This assignment operator is only valid if the @c OtherExecutor type is
   * convertible to @c Executor.
   */
  template <class OtherExecutor>
  strand& operator=(
      const strand<OtherExecutor>&& other) ASIO_NOEXCEPT
  {
    executor_ = ASIO_MOVE_CAST(OtherExecutor)(other);
    impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor.
  ~strand()
  {
  }

  /// Obtain the underlying executor.
  inner_executor_type get_inner_executor() const ASIO_NOEXCEPT
  {
    return executor_;
  }

  /// Obtain the underlying execution context.
  execution_context& context() const ASIO_NOEXCEPT
  {
    return executor_.context();
  }

  /// Inform the strand that it has some outstanding work to do.
  /**
   * The strand delegates this call to its underlying executor.
   */
  void on_work_started() const ASIO_NOEXCEPT
  {
    executor_.on_work_started();
  }

  /// Inform the strand that some work is no longer outstanding.
  /**
   * The strand delegates this call to its underlying executor.
   */
  void on_work_finished() const ASIO_NOEXCEPT
  {
    executor_.on_work_finished();
  }

  /// Request the strand to invoke the given function object.
  /**
   * This function is used to ask the strand to execute the given function
   * object on its underlying executor. The function object will be executed
   * inside this function if the strand is not otherwise busy and if the
   * underlying executor's @c dispatch() function is also able to execute the
   * function before returning.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void dispatch(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    detail::strand_executor_service::dispatch(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f), a);
  }

  /// Request the strand to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will never be executed inside this function.
   * Instead, it will be scheduled by the underlying executor's defer function.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void post(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    detail::strand_executor_service::post(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f), a);
  }

  /// Request the strand to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will never be executed inside this function.
   * Instead, it will be scheduled by the underlying executor's defer function.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void defer(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
  {
    detail::strand_executor_service::defer(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f), a);
  }

  /// Determine whether the strand is running in the current thread.
  /**
   * @return @c true if the current thread is executing a function that was
   * submitted to the strand using post(), dispatch() or defer(). Otherwise
   * returns @c false.
   */
  bool running_in_this_thread() const ASIO_NOEXCEPT
  {
    return detail::strand_executor_service::running_in_this_thread(impl_);
  }

  /// Compare two strands for equality.
  /**
   * Two strands are equal if they refer to the same ordered, non-concurrent
   * state.
   */
  friend bool operator==(const strand& a, const strand& b) ASIO_NOEXCEPT
  {
    return a.impl_ == b.impl_;
  }

  /// Compare two strands for inequality.
  /**
   * Two strands are equal if they refer to the same ordered, non-concurrent
   * state.
   */
  friend bool operator!=(const strand& a, const strand& b) ASIO_NOEXCEPT
  {
    return a.impl_ != b.impl_;
  }

private:
  Executor executor_;
  typedef detail::strand_executor_service::implementation_type
    implementation_type;
  implementation_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

// If both io_context.hpp and strand.hpp have been included, automatically
// include the header file needed for the io_context::strand class.
#if !defined(ASIO_NO_EXTENSIONS)
# if defined(ASIO_IO_CONTEXT_HPP)
#  include "asio/io_context_strand.hpp"
# endif // defined(ASIO_IO_CONTEXT_HPP)
#endif // !defined(ASIO_NO_EXTENSIONS)

#endif // ASIO_STRAND_HPP
