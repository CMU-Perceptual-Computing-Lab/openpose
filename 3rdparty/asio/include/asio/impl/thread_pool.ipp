//
// impl/thread_pool.ipp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_THREAD_POOL_IPP
#define ASIO_IMPL_THREAD_POOL_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/thread_pool.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

struct thread_pool::thread_function
{
  detail::scheduler* scheduler_;

  void operator()()
  {
    asio::error_code ec;
    scheduler_->run(ec);
  }
};

thread_pool::thread_pool()
  : scheduler_(use_service<detail::scheduler>(*this))
{
  scheduler_.work_started();

  thread_function f = { &scheduler_ };
  std::size_t num_threads = detail::thread::hardware_concurrency() * 2;
  threads_.create_threads(f, num_threads ? num_threads : 2);
}

thread_pool::thread_pool(std::size_t num_threads)
  : scheduler_(use_service<detail::scheduler>(*this))
{
  scheduler_.work_started();

  thread_function f = { &scheduler_ };
  threads_.create_threads(f, num_threads);
}

thread_pool::~thread_pool()
{
  stop();
  join();
}

void thread_pool::stop()
{
  scheduler_.stop();
}

void thread_pool::join()
{
  scheduler_.work_finished();
  threads_.join();
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_THREAD_POOL_IPP
