//
// detail/impl/win_object_handle_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2011 Boris Schaeling (boris@highscore.de)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WIN_OBJECT_HANDLE_SERVICE_IPP
#define ASIO_DETAIL_IMPL_WIN_OBJECT_HANDLE_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)

#include "asio/detail/win_object_handle_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

win_object_handle_service::win_object_handle_service(
    asio::io_context& io_context)
  : service_base<win_object_handle_service>(io_context),
    io_context_(asio::use_service<io_context_impl>(io_context)),
    mutex_(),
    impl_list_(0),
    shutdown_(false)
{
}

void win_object_handle_service::shutdown()
{
  mutex::scoped_lock lock(mutex_);

  // Setting this flag to true prevents new objects from being registered, and
  // new asynchronous wait operations from being started. We only need to worry
  // about cleaning up the operations that are currently in progress.
  shutdown_ = true;

  op_queue<operation> ops;
  for (implementation_type* impl = impl_list_; impl; impl = impl->next_)
    ops.push(impl->op_queue_);

  lock.unlock();

  io_context_.abandon_operations(ops);
}

void win_object_handle_service::construct(
    win_object_handle_service::implementation_type& impl)
{
  impl.handle_ = INVALID_HANDLE_VALUE;
  impl.wait_handle_ = INVALID_HANDLE_VALUE;
  impl.owner_ = this;

  // Insert implementation into linked list of all implementations.
  mutex::scoped_lock lock(mutex_);
  if (!shutdown_)
  {
    impl.next_ = impl_list_;
    impl.prev_ = 0;
    if (impl_list_)
      impl_list_->prev_ = &impl;
    impl_list_ = &impl;
  }
}

void win_object_handle_service::move_construct(
    win_object_handle_service::implementation_type& impl,
    win_object_handle_service::implementation_type& other_impl)
{
  mutex::scoped_lock lock(mutex_);

  // Insert implementation into linked list of all implementations.
  if (!shutdown_)
  {
    impl.next_ = impl_list_;
    impl.prev_ = 0;
    if (impl_list_)
      impl_list_->prev_ = &impl;
    impl_list_ = &impl;
  }

  impl.handle_ = other_impl.handle_;
  other_impl.handle_ = INVALID_HANDLE_VALUE;
  impl.wait_handle_ = other_impl.wait_handle_;
  other_impl.wait_handle_ = INVALID_HANDLE_VALUE;
  impl.op_queue_.push(other_impl.op_queue_);
  impl.owner_ = this;

  // We must not hold the lock while calling UnregisterWaitEx. This is because
  // the registered callback function might be invoked while we are waiting for
  // UnregisterWaitEx to complete.
  lock.unlock();

  if (impl.wait_handle_ != INVALID_HANDLE_VALUE)
    ::UnregisterWaitEx(impl.wait_handle_, INVALID_HANDLE_VALUE);

  if (!impl.op_queue_.empty())
    register_wait_callback(impl, lock);
}

void win_object_handle_service::move_assign(
    win_object_handle_service::implementation_type& impl,
    win_object_handle_service& other_service,
    win_object_handle_service::implementation_type& other_impl)
{
  asio::error_code ignored_ec;
  close(impl, ignored_ec);

  mutex::scoped_lock lock(mutex_);

  if (this != &other_service)
  {
    // Remove implementation from linked list of all implementations.
    if (impl_list_ == &impl)
      impl_list_ = impl.next_;
    if (impl.prev_)
      impl.prev_->next_ = impl.next_;
    if (impl.next_)
      impl.next_->prev_= impl.prev_;
    impl.next_ = 0;
    impl.prev_ = 0;
  }

  impl.handle_ = other_impl.handle_;
  other_impl.handle_ = INVALID_HANDLE_VALUE;
  impl.wait_handle_ = other_impl.wait_handle_;
  other_impl.wait_handle_ = INVALID_HANDLE_VALUE;
  impl.op_queue_.push(other_impl.op_queue_);
  impl.owner_ = this;

  if (this != &other_service)
  {
    // Insert implementation into linked list of all implementations.
    impl.next_ = other_service.impl_list_;
    impl.prev_ = 0;
    if (other_service.impl_list_)
      other_service.impl_list_->prev_ = &impl;
    other_service.impl_list_ = &impl;
  }

  // We must not hold the lock while calling UnregisterWaitEx. This is because
  // the registered callback function might be invoked while we are waiting for
  // UnregisterWaitEx to complete.
  lock.unlock();

  if (impl.wait_handle_ != INVALID_HANDLE_VALUE)
    ::UnregisterWaitEx(impl.wait_handle_, INVALID_HANDLE_VALUE);

  if (!impl.op_queue_.empty())
    register_wait_callback(impl, lock);
}

void win_object_handle_service::destroy(
    win_object_handle_service::implementation_type& impl)
{
  mutex::scoped_lock lock(mutex_);

  // Remove implementation from linked list of all implementations.
  if (impl_list_ == &impl)
    impl_list_ = impl.next_;
  if (impl.prev_)
    impl.prev_->next_ = impl.next_;
  if (impl.next_)
    impl.next_->prev_= impl.prev_;
  impl.next_ = 0;
  impl.prev_ = 0;

  if (is_open(impl))
  {
    ASIO_HANDLER_OPERATION((io_context_.context(), "object_handle",
          &impl, reinterpret_cast<uintmax_t>(impl.wait_handle_), "close"));

    HANDLE wait_handle = impl.wait_handle_;
    impl.wait_handle_ = INVALID_HANDLE_VALUE;

    op_queue<operation> ops;
    while (wait_op* op = impl.op_queue_.front())
    {
      op->ec_ = asio::error::operation_aborted;
      impl.op_queue_.pop();
      ops.push(op);
    }

    // We must not hold the lock while calling UnregisterWaitEx. This is
    // because the registered callback function might be invoked while we are
    // waiting for UnregisterWaitEx to complete.
    lock.unlock();

    if (wait_handle != INVALID_HANDLE_VALUE)
      ::UnregisterWaitEx(wait_handle, INVALID_HANDLE_VALUE);

    ::CloseHandle(impl.handle_);
    impl.handle_ = INVALID_HANDLE_VALUE;

    io_context_.post_deferred_completions(ops);
  }
}

asio::error_code win_object_handle_service::assign(
    win_object_handle_service::implementation_type& impl,
    const native_handle_type& handle, asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  impl.handle_ = handle;
  ec = asio::error_code();
  return ec;
}

asio::error_code win_object_handle_service::close(
    win_object_handle_service::implementation_type& impl,
    asio::error_code& ec)
{
  if (is_open(impl))
  {
    ASIO_HANDLER_OPERATION((io_context_.context(), "object_handle",
          &impl, reinterpret_cast<uintmax_t>(impl.wait_handle_), "close"));

    mutex::scoped_lock lock(mutex_);

    HANDLE wait_handle = impl.wait_handle_;
    impl.wait_handle_ = INVALID_HANDLE_VALUE;

    op_queue<operation> completed_ops;
    while (wait_op* op = impl.op_queue_.front())
    {
      impl.op_queue_.pop();
      op->ec_ = asio::error::operation_aborted;
      completed_ops.push(op);
    }

    // We must not hold the lock while calling UnregisterWaitEx. This is
    // because the registered callback function might be invoked while we are
    // waiting for UnregisterWaitEx to complete.
    lock.unlock();

    if (wait_handle != INVALID_HANDLE_VALUE)
      ::UnregisterWaitEx(wait_handle, INVALID_HANDLE_VALUE);

    if (::CloseHandle(impl.handle_))
    {
      impl.handle_ = INVALID_HANDLE_VALUE;
      ec = asio::error_code();
    }
    else
    {
      DWORD last_error = ::GetLastError();
      ec = asio::error_code(last_error,
          asio::error::get_system_category());
    }

    io_context_.post_deferred_completions(completed_ops);
  }
  else
  {
    ec = asio::error_code();
  }

  return ec;
}

asio::error_code win_object_handle_service::cancel(
    win_object_handle_service::implementation_type& impl,
    asio::error_code& ec)
{
  if (is_open(impl))
  {
    ASIO_HANDLER_OPERATION((io_context_.context(), "object_handle",
          &impl, reinterpret_cast<uintmax_t>(impl.wait_handle_), "cancel"));

    mutex::scoped_lock lock(mutex_);

    HANDLE wait_handle = impl.wait_handle_;
    impl.wait_handle_ = INVALID_HANDLE_VALUE;

    op_queue<operation> completed_ops;
    while (wait_op* op = impl.op_queue_.front())
    {
      op->ec_ = asio::error::operation_aborted;
      impl.op_queue_.pop();
      completed_ops.push(op);
    }

    // We must not hold the lock while calling UnregisterWaitEx. This is
    // because the registered callback function might be invoked while we are
    // waiting for UnregisterWaitEx to complete.
    lock.unlock();

    if (wait_handle != INVALID_HANDLE_VALUE)
      ::UnregisterWaitEx(wait_handle, INVALID_HANDLE_VALUE);

    ec = asio::error_code();

    io_context_.post_deferred_completions(completed_ops);
  }
  else
  {
    ec = asio::error::bad_descriptor;
  }

  return ec;
}

void win_object_handle_service::wait(
    win_object_handle_service::implementation_type& impl,
    asio::error_code& ec)
{
  switch (::WaitForSingleObject(impl.handle_, INFINITE))
  {
  case WAIT_FAILED:
    {
      DWORD last_error = ::GetLastError();
      ec = asio::error_code(last_error,
          asio::error::get_system_category());
      break;
    }
  case WAIT_OBJECT_0:
  case WAIT_ABANDONED:
  default:
    ec = asio::error_code();
    break;
  }
}

void win_object_handle_service::start_wait_op(
    win_object_handle_service::implementation_type& impl, wait_op* op)
{
  io_context_.work_started();

  if (is_open(impl))
  {
    mutex::scoped_lock lock(mutex_);

    if (!shutdown_)
    {
      impl.op_queue_.push(op);

      // Only the first operation to be queued gets to register a wait callback.
      // Subsequent operations have to wait for the first to finish.
      if (impl.op_queue_.front() == op)
        register_wait_callback(impl, lock);
    }
    else
    {
      lock.unlock();
      io_context_.post_deferred_completion(op);
    }
  }
  else
  {
    op->ec_ = asio::error::bad_descriptor;
    io_context_.post_deferred_completion(op);
  }
}

void win_object_handle_service::register_wait_callback(
    win_object_handle_service::implementation_type& impl,
    mutex::scoped_lock& lock)
{
  lock.lock();

  if (!RegisterWaitForSingleObject(&impl.wait_handle_,
        impl.handle_, &win_object_handle_service::wait_callback,
        &impl, INFINITE, WT_EXECUTEONLYONCE))
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());

    op_queue<operation> completed_ops;
    while (wait_op* op = impl.op_queue_.front())
    {
      op->ec_ = ec;
      impl.op_queue_.pop();
      completed_ops.push(op);
    }

    lock.unlock();
    io_context_.post_deferred_completions(completed_ops);
  }
}

void win_object_handle_service::wait_callback(PVOID param, BOOLEAN)
{
  implementation_type* impl = static_cast<implementation_type*>(param);
  mutex::scoped_lock lock(impl->owner_->mutex_);

  if (impl->wait_handle_ != INVALID_HANDLE_VALUE)
  {
    ::UnregisterWaitEx(impl->wait_handle_, NULL);
    impl->wait_handle_ = INVALID_HANDLE_VALUE;
  }

  if (wait_op* op = impl->op_queue_.front())
  {
    op_queue<operation> completed_ops;

    op->ec_ = asio::error_code();
    impl->op_queue_.pop();
    completed_ops.push(op);

    if (!impl->op_queue_.empty())
    {
      if (!RegisterWaitForSingleObject(&impl->wait_handle_,
            impl->handle_, &win_object_handle_service::wait_callback,
            param, INFINITE, WT_EXECUTEONLYONCE))
      {
        DWORD last_error = ::GetLastError();
        asio::error_code ec(last_error,
            asio::error::get_system_category());

        while ((op = impl->op_queue_.front()) != 0)
        {
          op->ec_ = ec;
          impl->op_queue_.pop();
          completed_ops.push(op);
        }
      }
    }

    io_context_impl& ioc = impl->owner_->io_context_;
    lock.unlock();
    ioc.post_deferred_completions(completed_ops);
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)

#endif // ASIO_DETAIL_IMPL_WIN_OBJECT_HANDLE_SERVICE_IPP
