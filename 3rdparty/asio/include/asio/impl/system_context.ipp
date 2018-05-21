//
// impl/system_context.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_SYSTEM_CONTEXT_IPP
#define ASIO_IMPL_SYSTEM_CONTEXT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/system_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

struct system_context::thread_function
{
  detail::scheduler* scheduler_;

  void operator()()
  {
    asio::error_code ec;
    scheduler_->run(ec);
  }
};

system_context::system_context()
  : scheduler_(use_service<detail::scheduler>(*this))
{
  scheduler_.work_started();

  thread_function f = { &scheduler_ };
  std::size_t num_threads = detail::thread::hardware_concurrency() * 2;
  threads_.create_threads(f, num_threads ? num_threads : 2);
}

system_context::~system_context()
{
  scheduler_.work_finished();
  scheduler_.stop();
  threads_.join();
}

void system_context::stop()
{
  scheduler_.stop();
}

bool system_context::stopped() const ASIO_NOEXCEPT
{
  return scheduler_.stopped();
}

void system_context::join()
{
  scheduler_.work_finished();
  threads_.join();
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_SYSTEM_CONTEXT_IPP
