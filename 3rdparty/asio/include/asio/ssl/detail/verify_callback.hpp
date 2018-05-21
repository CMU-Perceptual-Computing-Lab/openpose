//
// ssl/detail/verify_callback.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_VERIFY_CALLBACK_HPP
#define ASIO_SSL_DETAIL_VERIFY_CALLBACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/ssl/verify_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class verify_callback_base
{
public:
  virtual ~verify_callback_base()
  {
  }

  virtual bool call(bool preverified, verify_context& ctx) = 0;
};

template <typename VerifyCallback>
class verify_callback : public verify_callback_base
{
public:
  explicit verify_callback(VerifyCallback callback)
    : callback_(callback)
  {
  }

  virtual bool call(bool preverified, verify_context& ctx)
  {
    return callback_(preverified, ctx);
  }

private:
  VerifyCallback callback_;
};

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_VERIFY_CALLBACK_HPP
