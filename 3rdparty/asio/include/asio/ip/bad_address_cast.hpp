//
// ip/bad_address_cast.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_BAD_ADDRESS_CAST_HPP
#define ASIO_IP_BAD_ADDRESS_CAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <typeinfo>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Thrown to indicate a failed address conversion.
class bad_address_cast : public std::bad_cast
{
public:
  /// Default constructor.
  bad_address_cast() {}

  /// Destructor.
  virtual ~bad_address_cast() ASIO_NOEXCEPT_OR_NOTHROW {}

  /// Get the message associated with the exception.
  virtual const char* what() const ASIO_NOEXCEPT_OR_NOTHROW
  {
    return "bad address cast";
  }
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_ADDRESS_HPP
