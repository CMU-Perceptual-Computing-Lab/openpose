//
// ssl/impl/error.ipp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_IMPL_ERROR_IPP
#define ASIO_SSL_IMPL_ERROR_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/ssl/error.hpp"
#include "asio/ssl/detail/openssl_init.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace error {
namespace detail {

class ssl_category : public asio::error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.ssl";
  }

  std::string message(int value) const
  {
    const char* s = ::ERR_reason_error_string(value);
    return s ? s : "asio.ssl error";
  }
};

} // namespace detail

const asio::error_category& get_ssl_category()
{
  static detail::ssl_category instance;
  return instance;
}

} // namespace error
namespace ssl {
namespace error {

#if (OPENSSL_VERSION_NUMBER < 0x10100000L) && !defined(OPENSSL_IS_BORINGSSL)

const asio::error_category& get_stream_category()
{
  return asio::error::get_ssl_category();
}

#else

namespace detail {

class stream_category : public asio::error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.ssl.stream";
  }

  std::string message(int value) const
  {
    switch (value)
    {
    case stream_truncated: return "stream truncated";
    default: return "asio.ssl.stream error";
    }
  }
};

} // namespace detail

const asio::error_category& get_stream_category()
{
  static detail::stream_category instance;
  return instance;
}

#endif

} // namespace error
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_IMPL_ERROR_IPP
