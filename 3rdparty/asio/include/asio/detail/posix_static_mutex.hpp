//
// detail/posix_static_mutex.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_STATIC_MUTEX_HPP
#define ASIO_DETAIL_POSIX_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <pthread.h>
#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct posix_static_mutex
{
  typedef asio::detail::scoped_lock<posix_static_mutex> scoped_lock;

  // Initialise the mutex.
  void init()
  {
    // Nothing to do.
  }

  // Lock the mutex.
  void lock()
  {
    (void)::pthread_mutex_lock(&mutex_); // Ignore EINVAL.
  }

  // Unlock the mutex.
  void unlock()
  {
    (void)::pthread_mutex_unlock(&mutex_); // Ignore EINVAL.
  }

  ::pthread_mutex_t mutex_;
};

#define ASIO_POSIX_STATIC_MUTEX_INIT { PTHREAD_MUTEX_INITIALIZER }

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_PTHREADS)

#endif // ASIO_DETAIL_POSIX_STATIC_MUTEX_HPP
