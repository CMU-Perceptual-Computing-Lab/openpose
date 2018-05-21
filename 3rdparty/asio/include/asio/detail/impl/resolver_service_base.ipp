//
// detail/impl/resolver_service_base.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_RESOLVER_SERVICE_BASE_IPP
#define ASIO_DETAIL_IMPL_RESOLVER_SERVICE_BASE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/resolver_service_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class resolver_service_base::work_io_context_runner
{
public:
  work_io_context_runner(asio::io_context& io_context)
    : io_context_(io_context) {}
  void operator()() { io_context_.run(); }
private:
  asio::io_context& io_context_;
};

resolver_service_base::resolver_service_base(
    asio::io_context& io_context)
  : io_context_impl_(asio::use_service<io_context_impl>(io_context)),
    work_io_context_(new asio::io_context(-1)),
    work_io_context_impl_(asio::use_service<
        io_context_impl>(*work_io_context_)),
    work_(asio::make_work_guard(*work_io_context_)),
    work_thread_(0)
{
}

resolver_service_base::~resolver_service_base()
{
  base_shutdown();
}

void resolver_service_base::base_shutdown()
{
  work_.reset();
  if (work_io_context_.get())
  {
    work_io_context_->stop();
    if (work_thread_.get())
    {
      work_thread_->join();
      work_thread_.reset();
    }
    work_io_context_.reset();
  }
}

void resolver_service_base::base_notify_fork(
    asio::io_context::fork_event fork_ev)
{
  if (work_thread_.get())
  {
    if (fork_ev == asio::io_context::fork_prepare)
    {
      work_io_context_->stop();
      work_thread_->join();
    }
    else
    {
      work_io_context_->restart();
      work_thread_.reset(new asio::detail::thread(
            work_io_context_runner(*work_io_context_)));
    }
  }
}

void resolver_service_base::construct(
    resolver_service_base::implementation_type& impl)
{
  impl.reset(static_cast<void*>(0), socket_ops::noop_deleter());
}

void resolver_service_base::destroy(
    resolver_service_base::implementation_type& impl)
{
  ASIO_HANDLER_OPERATION((io_context_impl_.context(),
        "resolver", &impl, 0, "cancel"));

  impl.reset();
}

void resolver_service_base::move_construct(implementation_type& impl,
    implementation_type& other_impl)
{
  impl = ASIO_MOVE_CAST(implementation_type)(other_impl);
}

void resolver_service_base::move_assign(implementation_type& impl,
    resolver_service_base&, implementation_type& other_impl)
{
  destroy(impl);
  impl = ASIO_MOVE_CAST(implementation_type)(other_impl);
}

void resolver_service_base::cancel(
    resolver_service_base::implementation_type& impl)
{
  ASIO_HANDLER_OPERATION((io_context_impl_.context(),
        "resolver", &impl, 0, "cancel"));

  impl.reset(static_cast<void*>(0), socket_ops::noop_deleter());
}

void resolver_service_base::start_resolve_op(resolve_op* op)
{
  if (ASIO_CONCURRENCY_HINT_IS_LOCKING(SCHEDULER,
        io_context_impl_.concurrency_hint()))
  {
    start_work_thread();
    io_context_impl_.work_started();
    work_io_context_impl_.post_immediate_completion(op, false);
  }
  else
  {
    op->ec_ = asio::error::operation_not_supported;
    io_context_impl_.post_immediate_completion(op, false);
  }
}

void resolver_service_base::start_work_thread()
{
  asio::detail::mutex::scoped_lock lock(mutex_);
  if (!work_thread_.get())
  {
    work_thread_.reset(new asio::detail::thread(
          work_io_context_runner(*work_io_context_)));
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_RESOLVER_SERVICE_BASE_IPP
