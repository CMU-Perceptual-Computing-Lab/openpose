//
// executor.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTOR_HPP
#define ASIO_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <typeinfo>
#include "asio/detail/cstddef.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/throw_exception.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Exception thrown when trying to access an empty polymorphic executor.
class bad_executor
  : public std::exception
{
public:
  /// Constructor.
  ASIO_DECL bad_executor() ASIO_NOEXCEPT;

  /// Obtain message associated with exception.
  ASIO_DECL virtual const char* what() const
    ASIO_NOEXCEPT_OR_NOTHROW;
};

/// Polymorphic wrapper for executors.
class executor
{
public:
  /// Default constructor.
  executor() ASIO_NOEXCEPT
    : impl_(0)
  {
  }

  /// Construct from nullptr.
  executor(nullptr_t) ASIO_NOEXCEPT
    : impl_(0)
  {
  }

  /// Copy constructor.
  executor(const executor& other) ASIO_NOEXCEPT
    : impl_(other.clone())
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move constructor.
  executor(executor&& other) ASIO_NOEXCEPT
    : impl_(other.impl_)
  {
    other.impl_ = 0;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Construct a polymorphic wrapper for the specified executor.
  template <typename Executor>
  executor(Executor e);

  /// Allocator-aware constructor to create a polymorphic wrapper for the
  /// specified executor.
  template <typename Executor, typename Allocator>
  executor(allocator_arg_t, const Allocator& a, Executor e);

  /// Destructor.
  ~executor()
  {
    destroy();
  }

  /// Assignment operator.
  executor& operator=(const executor& other) ASIO_NOEXCEPT
  {
    destroy();
    impl_ = other.clone();
    return *this;
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  // Move assignment operator.
  executor& operator=(executor&& other) ASIO_NOEXCEPT
  {
    destroy();
    impl_ = other.impl_;
    other.impl_ = 0;
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Assignment operator for nullptr_t.
  executor& operator=(nullptr_t) ASIO_NOEXCEPT
  {
    destroy();
    impl_ = 0;
    return *this;
  }

  /// Assignment operator to create a polymorphic wrapper for the specified
  /// executor.
  template <typename Executor>
  executor& operator=(ASIO_MOVE_ARG(Executor) e) ASIO_NOEXCEPT
  {
    executor tmp(ASIO_MOVE_CAST(Executor)(e));
    destroy();
    impl_ = tmp.impl_;
    tmp.impl_ = 0;
    return *this;
  }

  /// Obtain the underlying execution context.
  execution_context& context() const ASIO_NOEXCEPT
  {
    return get_impl()->context();
  }

  /// Inform the executor that it has some outstanding work to do.
  void on_work_started() const ASIO_NOEXCEPT
  {
    get_impl()->on_work_started();
  }

  /// Inform the executor that some work is no longer outstanding.
  void on_work_finished() const ASIO_NOEXCEPT
  {
    get_impl()->on_work_finished();
  }

  /// Request the executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object is executed according to the rules of the
   * target executor object.
   *
   * @param f The function object to be called. The executor will make a copy
   * of the handler object as required. The function signature of the function
   * object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void dispatch(ASIO_MOVE_ARG(Function) f, const Allocator& a) const;

  /// Request the executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object is executed according to the rules of the
   * target executor object.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void post(ASIO_MOVE_ARG(Function) f, const Allocator& a) const;

  /// Request the executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object is executed according to the rules of the
   * target executor object.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   *
   * @param a An allocator that may be used by the executor to allocate the
   * internal storage needed for function invocation.
   */
  template <typename Function, typename Allocator>
  void defer(ASIO_MOVE_ARG(Function) f, const Allocator& a) const;

  struct unspecified_bool_type_t {};
  typedef void (*unspecified_bool_type)(unspecified_bool_type_t);
  static void unspecified_bool_true(unspecified_bool_type_t) {}

  /// Operator to test if the executor contains a valid target.
  operator unspecified_bool_type() const ASIO_NOEXCEPT
  {
    return impl_ ? &executor::unspecified_bool_true : 0;
  }

  /// Obtain type information for the target executor object.
  /**
   * @returns If @c *this has a target type of type @c T, <tt>typeid(T)</tt>;
   * otherwise, <tt>typeid(void)</tt>.
   */
#if !defined(ASIO_NO_TYPEID) || defined(GENERATING_DOCUMENTATION)
  const std::type_info& target_type() const ASIO_NOEXCEPT
  {
    return impl_ ? impl_->target_type() : typeid(void);
  }
#else // !defined(ASIO_NO_TYPEID) || defined(GENERATING_DOCUMENTATION)
  const void* target_type() const ASIO_NOEXCEPT
  {
    return impl_ ? impl_->target_type() : 0;
  }
#endif // !defined(ASIO_NO_TYPEID) || defined(GENERATING_DOCUMENTATION)

  /// Obtain a pointer to the target executor object.
  /**
   * @returns If <tt>target_type() == typeid(T)</tt>, a pointer to the stored
   * executor target; otherwise, a null pointer.
   */
  template <typename Executor>
  Executor* target() ASIO_NOEXCEPT;

  /// Obtain a pointer to the target executor object.
  /**
   * @returns If <tt>target_type() == typeid(T)</tt>, a pointer to the stored
   * executor target; otherwise, a null pointer.
   */
  template <typename Executor>
  const Executor* target() const ASIO_NOEXCEPT;

  /// Compare two executors for equality.
  friend bool operator==(const executor& a,
      const executor& b) ASIO_NOEXCEPT
  {
    if (a.impl_ == b.impl_)
      return true;
    if (!a.impl_ || !b.impl_)
      return false;
    return a.impl_->equals(b.impl_);
  }

  /// Compare two executors for inequality.
  friend bool operator!=(const executor& a,
      const executor& b) ASIO_NOEXCEPT
  {
    return !(a == b);
  }

private:
#if !defined(GENERATING_DOCUMENTATION)
  class function;
  template <typename, typename> class impl;

#if !defined(ASIO_NO_TYPEID)
  typedef const std::type_info& type_id_result_type;
#else // !defined(ASIO_NO_TYPEID)
  typedef const void* type_id_result_type;
#endif // !defined(ASIO_NO_TYPEID)

  template <typename T>
  static type_id_result_type type_id()
  {
#if !defined(ASIO_NO_TYPEID)
    return typeid(T);
#else // !defined(ASIO_NO_TYPEID)
    static int unique_id;
    return &unique_id;
#endif // !defined(ASIO_NO_TYPEID)
  }

  // Base class for all polymorphic executor implementations.
  class impl_base
  {
  public:
    virtual impl_base* clone() const ASIO_NOEXCEPT = 0;
    virtual void destroy() ASIO_NOEXCEPT = 0;
    virtual execution_context& context() ASIO_NOEXCEPT = 0;
    virtual void on_work_started() ASIO_NOEXCEPT = 0;
    virtual void on_work_finished() ASIO_NOEXCEPT = 0;
    virtual void dispatch(ASIO_MOVE_ARG(function)) = 0;
    virtual void post(ASIO_MOVE_ARG(function)) = 0;
    virtual void defer(ASIO_MOVE_ARG(function)) = 0;
    virtual type_id_result_type target_type() const ASIO_NOEXCEPT = 0;
    virtual void* target() ASIO_NOEXCEPT = 0;
    virtual const void* target() const ASIO_NOEXCEPT = 0;
    virtual bool equals(const impl_base* e) const ASIO_NOEXCEPT = 0;

  protected:
    impl_base(bool fast_dispatch) : fast_dispatch_(fast_dispatch) {}
    virtual ~impl_base() {}

  private:
    friend class executor;
    const bool fast_dispatch_;
  };

  // Helper function to check and return the implementation pointer.
  impl_base* get_impl() const
  {
    if (!impl_)
    {
      bad_executor ex;
      asio::detail::throw_exception(ex);
    }
    return impl_;
  }

  // Helper function to clone another implementation.
  impl_base* clone() const ASIO_NOEXCEPT
  {
    return impl_ ? impl_->clone() : 0;
  }

  // Helper function to destroy an implementation.
  void destroy() ASIO_NOEXCEPT
  {
    if (impl_)
      impl_->destroy();
  }

  impl_base* impl_;
#endif // !defined(GENERATING_DOCUMENTATION)
};

} // namespace asio

ASIO_USES_ALLOCATOR(asio::executor)

#include "asio/detail/pop_options.hpp"

#include "asio/impl/executor.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/executor.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_EXECUTOR_HPP
