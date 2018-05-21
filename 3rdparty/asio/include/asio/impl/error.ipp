//
// impl/error.ipp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ERROR_IPP
#define ASIO_IMPL_ERROR_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <string>
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace error {

#if !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)

namespace detail {

class netdb_category : public asio::error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.netdb";
  }

  std::string message(int value) const
  {
    if (value == error::host_not_found)
      return "Host not found (authoritative)";
    if (value == error::host_not_found_try_again)
      return "Host not found (non-authoritative), try again later";
    if (value == error::no_data)
      return "The query is valid, but it does not have associated data";
    if (value == error::no_recovery)
      return "A non-recoverable error occurred during database lookup";
    return "asio.netdb error";
  }
};

} // namespace detail

const asio::error_category& get_netdb_category()
{
  static detail::netdb_category instance;
  return instance;
}

namespace detail {

class addrinfo_category : public asio::error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.addrinfo";
  }

  std::string message(int value) const
  {
    if (value == error::service_not_found)
      return "Service not found";
    if (value == error::socket_type_not_supported)
      return "Socket type not supported";
    return "asio.addrinfo error";
  }
};

} // namespace detail

const asio::error_category& get_addrinfo_category()
{
  static detail::addrinfo_category instance;
  return instance;
}

#endif // !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)

namespace detail {

class misc_category : public asio::error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.misc";
  }

  std::string message(int value) const
  {
    if (value == error::already_open)
      return "Already open";
    if (value == error::eof)
      return "End of file";
    if (value == error::not_found)
      return "Element not found";
    if (value == error::fd_set_failure)
      return "The descriptor does not fit into the select call's fd_set";
    return "asio.misc error";
  }
};

} // namespace detail

const asio::error_category& get_misc_category()
{
  static detail::misc_category instance;
  return instance;
}

} // namespace error
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_ERROR_IPP
