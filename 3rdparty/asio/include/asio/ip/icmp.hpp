//
// ip/icmp.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_ICMP_HPP
#define ASIO_IP_ICMP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/basic_raw_socket.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Encapsulates the flags needed for ICMP.
/**
 * The asio::ip::icmp class contains flags necessary for ICMP sockets.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol, InternetProtocol.
 */
class icmp
{
public:
  /// The type of a ICMP endpoint.
  typedef basic_endpoint<icmp> endpoint;

  /// Construct to represent the IPv4 ICMP protocol.
  static icmp v4()
  {
    return icmp(ASIO_OS_DEF(IPPROTO_ICMP),
        ASIO_OS_DEF(AF_INET));
  }

  /// Construct to represent the IPv6 ICMP protocol.
  static icmp v6()
  {
    return icmp(ASIO_OS_DEF(IPPROTO_ICMPV6),
        ASIO_OS_DEF(AF_INET6));
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

  /// The ICMP socket type.
  typedef basic_raw_socket<icmp> socket;

  /// The ICMP resolver type.
  typedef basic_resolver<icmp> resolver;

  /// Compare two protocols for equality.
  friend bool operator==(const icmp& p1, const icmp& p2)
  {
    return p1.protocol_ == p2.protocol_ && p1.family_ == p2.family_;
  }

  /// Compare two protocols for inequality.
  friend bool operator!=(const icmp& p1, const icmp& p2)
  {
    return p1.protocol_ != p2.protocol_ || p1.family_ != p2.family_;
  }

private:
  // Construct with a specific family.
  explicit icmp(int protocol_id, int protocol_family)
    : protocol_(protocol_id),
      family_(protocol_family)
  {
  }

  int protocol_;
  int family_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_ICMP_HPP
