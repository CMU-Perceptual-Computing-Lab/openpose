//
// detail/impl/strand_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP
#define ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/strand_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct strand_service::on_do_complete_exit
{
  io_context_impl* owner_;
  strand_impl* impl_;

  ~on_do_complete_exit()
  {
    impl_->mutex_.lock();
    impl_->ready_queue_.push(impl_->waiting_queue_);
    bool more_handlers = impl_->locked_ = !impl_->ready_queue_.empty();
    impl_->mutex_.unlock();

    if (more_handlers)
      owner_->post_immediate_completion(impl_, true);
  }
};

strand_service::strand_service(asio::io_context& io_context)
  : asio::detail::service_base<strand_service>(io_context),
    io_context_(asio::use_service<io_context_impl>(io_context)),
    mutex_(),
    salt_(0)
{
}

void strand_service::shutdown()
{
  op_queue<operation> ops;

  asio::detail::mutex::scoped_lock lock(mutex_);

  for (std::size_t i = 0; i < num_implementations; ++i)
  {
    if (strand_impl* impl = implementations_[i].get())
    {
      ops.push(impl->waiting_queue_);
      ops.push(impl->ready_queue_);
    }
  }
}

void strand_service::construct(strand_service::implementation_type& impl)
{
  asio::detail::mutex::scoped_lock lock(mutex_);

  std::size_t salt = salt_++;
#if defined(ASIO_ENABLE_SEQUENTIAL_STRAND_ALLOCATION)
  std::size_t index = salt;
#else // defined(ASIO_ENABLE_SEQUENTIAL_STRAND_ALLOCATION)
  std::size_t index = reinterpret_cast<std::size_t>(&impl);
  index += (reinterpret_cast<std::size_t>(&impl) >> 3);
  index ^= salt + 0x9e3779b9 + (index << 6) + (index >> 2);
#endif // defined(ASIO_ENABLE_SEQUENTIAL_STRAND_ALLOCATION)
  index = index % num_implementations;

  if (!implementations_[index].get())
    implementations_[index].reset(new strand_impl);
  impl = implementations_[index].get();
}

bool strand_service::running_in_this_thread(
    const implementation_type& impl) const
{
  return call_stack<strand_impl>::contains(impl) != 0;
}

bool strand_service::do_dispatch(implementation_type& impl, operation* op)
{
  // If we are running inside the io_context, and no other handler already
  // holds the strand lock, then the handler can run immediately.
  bool can_dispatch = io_context_.can_dispatch();
  impl->mutex_.lock();
  if (can_dispatch && !impl->locked_)
  {
    // Immediate invocation is allowed.
    impl->locked_ = true;
    impl->mutex_.unlock();
    return true;
  }

  if (impl->locked_)
  {
    // Some other handler already holds the strand lock. Enqueue for later.
    impl->waiting_queue_.push(op);
    impl->mutex_.unlock();
  }
  else
  {
    // The handler is acquiring the strand lock and so is responsible for
    // scheduling the strand.
    impl->locked_ = true;
    impl->mutex_.unlock();
    impl->ready_queue_.push(op);
    io_context_.post_immediate_completion(impl, false);
  }

  return false;
}

void strand_service::do_post(implementation_type& impl,
    operation* op, bool is_continuation)
{
  impl->mutex_.lock();
  if (impl->locked_)
  {
    // Some other handler already holds the strand lock. Enqueue for later.
    impl->waiting_queue_.push(op);
    impl->mutex_.unlock();
  }
  else
  {
    // The handler is acquiring the strand lock and so is responsible for
    // scheduling the strand.
    impl->locked_ = true;
    impl->mutex_.unlock();
    impl->ready_queue_.push(op);
    io_context_.post_immediate_completion(impl, is_continuation);
  }
}

void strand_service::do_complete(void* owner, operation* base,
    const asio::error_code& ec, std::size_t /*bytes_transferred*/)
{
  if (owner)
  {
    strand_impl* impl = static_cast<strand_impl*>(base);

    // Indicate that this strand is executing on the current thread.
    call_stack<strand_impl>::context ctx(impl);

    // Ensure the next handler, if any, is scheduled on block exit.
    on_do_complete_exit on_exit;
    on_exit.owner_ = static_cast<io_context_impl*>(owner);
    on_exit.impl_ = impl;

    // Run all ready handlers. No lock is required since the ready queue is
    // accessed only within the strand.
    while (operation* o = impl->ready_queue_.front())
    {
      impl->ready_queue_.pop();
      o->complete(owner, ec, 0);
    }
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP
