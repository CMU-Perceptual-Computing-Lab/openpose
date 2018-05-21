//
// detail/impl/posix_event.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_EVENT_IPP
#define ASIO_DETAIL_IMPL_POSIX_EVENT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include "asio/detail/posix_event.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

posix_event::posix_event()
  : state_(0)
{
#if (defined(__MACH__) && defined(__APPLE__)) \
      || (defined(__ANDROID__) && (__ANDROID_API__ < 21))
  int error = ::pthread_cond_init(&cond_, 0);
#else // (defined(__MACH__) && defined(__APPLE__))
      // || (defined(__ANDROID__) && (__ANDROID_API__ < 21))
  ::pthread_condattr_t attr;
  ::pthread_condattr_init(&attr);
  int error = ::pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
  if (error == 0)
    error = ::pthread_cond_init(&cond_, &attr);
#endif // (defined(__MACH__) && defined(__APPLE__))
       // || (defined(__ANDROID__) && (__ANDROID_API__ < 21))

  asio::error_code ec(error,
      asio::error::get_system_category());
  asio::detail::throw_error(ec, "event");
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_PTHREADS)

#endif // ASIO_DETAIL_IMPL_POSIX_EVENT_IPP
