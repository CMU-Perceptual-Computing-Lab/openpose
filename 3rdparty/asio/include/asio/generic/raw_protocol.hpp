//
// generic/raw_protocol.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GENERIC_RAW_PROTOCOL_HPP
#define ASIO_GENERIC_RAW_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <typeinfo>
#include "asio/basic_raw_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_exception.hpp"
#include "asio/generic/basic_endpoint.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace generic {

/// Encapsulates the flags needed for a generic raw socket.
/**
 * The asio::generic::raw_protocol class contains flags necessary for
 * raw sockets of any address family and protocol.
 *
 * @par Examples
 * Constructing using a native address family and socket protocol:
 * @code raw_protocol p(AF_INET, IPPROTO_ICMP); @endcode
 * Constructing from a specific protocol type:
 * @code raw_protocol p(asio::ip::icmp::v4()); @endcode
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol.
 */
class raw_protocol
{
public:
  /// Construct a protocol object for a specific address family and protocol.
  raw_protocol(int address_family, int socket_protocol)
    : family_(address_family),
      protocol_(socket_protocol)
  {
  }

  /// Construct a generic protocol object from a specific protocol.
  /**
   * @throws @c bad_cast Thrown if the source protocol is not raw-oriented.
   */
  template <typename Protocol>
  raw_protocol(const Protocol& source_protocol)
    : family_(source_protocol.family()),
      protocol_(source_protocol.protocol())
  {
    if (source_protocol.type() != type())
    {
      std::bad_cast ex;
      asio::detail::throw_exception(ex);
    }
  }

  /// Obtain an identifier for the type of the protocol.
  int type() const
  {
    return ASIO_OS_DEF(SOCK_RAW);
  }

  /// Obtain an identifier for the protocol.
  int protocol() const
  {
    return protocol_;
  }

  /// Obtain an identifier for the protocol family.
  int family() const
  {
    return family_;
  }

  /// Compare two protocols for equality.
  friend bool operator==(const raw_protocol& p1, const raw_protocol& p2)
  {
    return p1.family_ == p2.family_ && p1.protocol_ == p2.protocol_;
  }

  /// Compare two protocols for inequality.
  friend bool operator!=(const raw_protocol& p1, const raw_protocol& p2)
  {
    return !(p1 == p2);
  }

  /// The type of an endpoint.
  typedef basic_endpoint<raw_protocol> endpoint;

  /// The generic socket type.
  typedef basic_raw_socket<raw_protocol> socket;

private:
  int family_;
  int protocol_;
};

} // namespace generic
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_GENERIC_RAW_PROTOCOL_HPP
