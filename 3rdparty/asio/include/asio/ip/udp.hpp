//
// ip/udp.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_UDP_HPP
#define ASIO_IP_UDP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/basic_datagram_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Encapsulates the flags needed for UDP.
/**
 * The asio::ip::udp class contains flags necessary for UDP sockets.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol, InternetProtocol.
 */
class udp
{
public:
  /// The type of a UDP endpoint.
  typedef basic_endpoint<udp> endpoint;

  /// Construct to represent the IPv4 UDP protocol.
  static udp v4()
  {
    return udp(ASIO_OS_DEF(AF_INET));
  }

  /// Construct to represent the IPv6 UDP protocol.
  static udp v6()
  {
    return udp(ASIO_OS_DEF(AF_INET6));
  }

  /// Obtain an identifier for the type of the protocol.
  int type() const
  {
    return ASIO_OS_DEF(SOCK_DGRAM);
  }

  /// Obtain an identifier for the protocol.
  int protocol() const
  {
    return ASIO_OS_DEF(IPPROTO_UDP);
  }

  /// Obtain an identifier for the protocol family.
  int family() const
  {
    return family_;
  }

  /// The UDP socket type.
  typedef basic_datagram_socket<udp> socket;

  /// The UDP resolver type.
  typedef basic_resolver<udp> resolver;

  /// Compare two protocols for equality.
  friend bool operator==(const udp& p1, const udp& p2)
  {
    return p1.family_ == p2.family_;
  }

  /// Compare two protocols for inequality.
  friend bool operator!=(const udp& p1, const udp& p2)
  {
    return p1.family_ != p2.family_;
  }

private:
  // Construct with a specific family.
  explicit udp(int protocol_family)
    : family_(protocol_family)
  {
  }

  int family_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_UDP_HPP
