//
// impl/io_context.ipp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_IO_CONTEXT_IPP
#define ASIO_IMPL_IO_CONTEXT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/io_context.hpp"
#include "asio/detail/concurrency_hint.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/throw_error.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else
# include "asio/detail/scheduler.hpp"
#endif

#include "asio/detail/push_options.hpp"

namespace asio {

io_context::io_context()
  : impl_(add_impl(new impl_type(*this, ASIO_CONCURRENCY_HINT_DEFAULT)))
{
}

io_context::io_context(int concurrency_hint)
  : impl_(add_impl(new impl_type(*this, concurrency_hint == 1
          ? ASIO_CONCURRENCY_HINT_1 : concurrency_hint)))
{
}

io_context::impl_type& io_context::add_impl(io_context::impl_type* impl)
{
  asio::detail::scoped_ptr<impl_type> scoped_impl(impl);
  asio::add_service<impl_type>(*this, scoped_impl.get());
  return *scoped_impl.release();
}

io_context::~io_context()
{
}

io_context::count_type io_context::run()
{
  asio::error_code ec;
  count_type s = impl_.run(ec);
  asio::detail::throw_error(ec);
  return s;
}

#if !defined(ASIO_NO_DEPRECATED)
io_context::count_type io_context::run(asio::error_code& ec)
{
  return impl_.run(ec);
}
#endif // !defined(ASIO_NO_DEPRECATED)

io_context::count_type io_context::run_one()
{
  asio::error_code ec;
  count_type s = impl_.run_one(ec);
  asio::detail::throw_error(ec);
  return s;
}

#if !defined(ASIO_NO_DEPRECATED)
io_context::count_type io_context::run_one(asio::error_code& ec)
{
  return impl_.run_one(ec);
}
#endif // !defined(ASIO_NO_DEPRECATED)

io_context::count_type io_context::poll()
{
  asio::error_code ec;
  count_type s = impl_.poll(ec);
  asio::detail::throw_error(ec);
  return s;
}

#if !defined(ASIO_NO_DEPRECATED)
io_context::count_type io_context::poll(asio::error_code& ec)
{
  return impl_.poll(ec);
}
#endif // !defined(ASIO_NO_DEPRECATED)

io_context::count_type io_context::poll_one()
{
  asio::error_code ec;
  count_type s = impl_.poll_one(ec);
  asio::detail::throw_error(ec);
  return s;
}

#if !defined(ASIO_NO_DEPRECATED)
io_context::count_type io_context::poll_one(asio::error_code& ec)
{
  return impl_.poll_one(ec);
}
#endif // !defined(ASIO_NO_DEPRECATED)

void io_context::stop()
{
  impl_.stop();
}

bool io_context::stopped() const
{
  return impl_.stopped();
}

void io_context::restart()
{
  impl_.restart();
}

io_context::service::service(asio::io_context& owner)
  : execution_context::service(owner)
{
}

io_context::service::~service()
{
}

void io_context::service::shutdown()
{
#if !defined(ASIO_NO_DEPRECATED)
  shutdown_service();
#endif // !defined(ASIO_NO_DEPRECATED)
}

#if !defined(ASIO_NO_DEPRECATED)
void io_context::service::shutdown_service()
{
}
#endif // !defined(ASIO_NO_DEPRECATED)

void io_context::service::notify_fork(io_context::fork_event ev)
{
#if !defined(ASIO_NO_DEPRECATED)
  fork_service(ev);
#else // !defined(ASIO_NO_DEPRECATED)
  (void)ev;
#endif // !defined(ASIO_NO_DEPRECATED)
}

#if !defined(ASIO_NO_DEPRECATED)
void io_context::service::fork_service(io_context::fork_event)
{
}
#endif // !defined(ASIO_NO_DEPRECATED)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_IO_CONTEXT_IPP
