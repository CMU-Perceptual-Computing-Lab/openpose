//
// detail/impl/winsock_init.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WINSOCK_INIT_IPP
#define ASIO_DETAIL_IMPL_WINSOCK_INIT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)

#include "asio/detail/socket_types.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

void winsock_init_base::startup(data& d,
    unsigned char major, unsigned char minor)
{
  if (::InterlockedIncrement(&d.init_count_) == 1)
  {
    WSADATA wsa_data;
    long result = ::WSAStartup(MAKEWORD(major, minor), &wsa_data);
    ::InterlockedExchange(&d.result_, result);
  }
}

void winsock_init_base::manual_startup(data& d)
{
  if (::InterlockedIncrement(&d.init_count_) == 1)
  {
    ::InterlockedExchange(&d.result_, 0);
  }
}

void winsock_init_base::cleanup(data& d)
{
  if (::InterlockedDecrement(&d.init_count_) == 0)
  {
    ::WSACleanup();
  }
}

void winsock_init_base::manual_cleanup(data& d)
{
  ::InterlockedDecrement(&d.init_count_);
}

void winsock_init_base::throw_on_error(data& d)
{
  long result = ::InterlockedExchangeAdd(&d.result_, 0);
  if (result != 0)
  {
    asio::error_code ec(result,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "winsock");
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

#endif // ASIO_DETAIL_IMPL_WINSOCK_INIT_IPP
