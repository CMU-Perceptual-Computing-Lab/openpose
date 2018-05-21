//
// ip/network_v4.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2014 Oliver Kowalke (oliver dot kowalke at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_NETWORK_V4_HPP
#define ASIO_IP_NETWORK_V4_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <string>
#include "asio/detail/string_view.hpp"
#include "asio/error_code.hpp"
#include "asio/ip/address_v4_range.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Represents an IPv4 network.
/**
 * The asio::ip::network_v4 class provides the ability to use and
 * manipulate IP version 4 networks.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class network_v4
{
public:
  /// Default constructor.
  network_v4() ASIO_NOEXCEPT
    : address_(),
      prefix_length_(0)
  {
  }

  /// Construct a network based on the specified address and prefix length.
  ASIO_DECL network_v4(const address_v4& addr,
      unsigned short prefix_len);

  /// Construct network based on the specified address and netmask.
  ASIO_DECL network_v4(const address_v4& addr,
      const address_v4& mask);

  /// Copy constructor.
  network_v4(const network_v4& other) ASIO_NOEXCEPT
    : address_(other.address_),
      prefix_length_(other.prefix_length_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  /// Move constructor.
  network_v4(network_v4&& other) ASIO_NOEXCEPT
    : address_(ASIO_MOVE_CAST(address_v4)(other.address_)),
      prefix_length_(other.prefix_length_)
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Assign from another network.
  network_v4& operator=(const network_v4& other) ASIO_NOEXCEPT
  {
    address_ = other.address_;
    prefix_length_ = other.prefix_length_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE)
  /// Move-assign from another network.
  network_v4& operator=(network_v4&& other) ASIO_NOEXCEPT
  {
    address_ = ASIO_MOVE_CAST(address_v4)(other.address_);
    prefix_length_ = other.prefix_length_;
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Obtain the address object specified when the network object was created.
  address_v4 address() const ASIO_NOEXCEPT
  {
    return address_;
  }

  /// Obtain the prefix length that was specified when the network object was
  /// created.
  unsigned short prefix_length() const ASIO_NOEXCEPT
  {
    return prefix_length_;
  }

  /// Obtain the netmask that was specified when the network object was created.
  ASIO_DECL address_v4 netmask() const ASIO_NOEXCEPT;

  /// Obtain an address object that represents the network address.
  address_v4 network() const ASIO_NOEXCEPT
  {
    return address_v4(address_.to_uint() & netmask().to_uint());
  }

  /// Obtain an address object that represents the network's broadcast address.
  address_v4 broadcast() const ASIO_NOEXCEPT
  {
    return address_v4(network().to_uint() | (netmask().to_uint() ^ 0xFFFFFFFF));
  }

  /// Obtain an address range corresponding to the hosts in the network.
  ASIO_DECL address_v4_range hosts() const ASIO_NOEXCEPT;

  /// Obtain the true network address, omitting any host bits.
  network_v4 canonical() const ASIO_NOEXCEPT
  {
    return network_v4(network(), netmask());
  }

  /// Test if network is a valid host address.
  bool is_host() const ASIO_NOEXCEPT
  {
    return prefix_length_ == 32;
  }

  /// Test if a network is a real subnet of another network.
  ASIO_DECL bool is_subnet_of(const network_v4& other) const;

  /// Get the network as an address in dotted decimal format.
  ASIO_DECL std::string to_string() const;

  /// Get the network as an address in dotted decimal format.
  ASIO_DECL std::string to_string(asio::error_code& ec) const;

  /// Compare two networks for equality.
  friend bool operator==(const network_v4& a, const network_v4& b)
  {
    return a.address_ == b.address_ && a.prefix_length_ == b.prefix_length_;
  }

  /// Compare two networks for inequality.
  friend bool operator!=(const network_v4& a, const network_v4& b)
  {
    return !(a == b);
  }

private:
  address_v4 address_;
  unsigned short prefix_length_;
};

/// Create an IPv4 network from an address and prefix length.
/**
 * @relates address_v4
 */
inline network_v4 make_network_v4(
    const address_v4& addr, unsigned short prefix_len)
{
  return network_v4(addr, prefix_len);
}

/// Create an IPv4 network from an address and netmask.
/**
 * @relates address_v4
 */
inline network_v4 make_network_v4(
    const address_v4& addr, const address_v4& mask)
{
  return network_v4(addr, mask);
}

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(const char* str);

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(
    const char* str, asio::error_code& ec);

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(const std::string& str);

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(
    const std::string& str, asio::error_code& ec);

#if defined(ASIO_HAS_STRING_VIEW) \
  || defined(GENERATING_DOCUMENTATION)

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(string_view str);

/// Create an IPv4 network from a string containing IP address and prefix
/// length.
/**
 * @relates network_v4
 */
ASIO_DECL network_v4 make_network_v4(
    string_view str, asio::error_code& ec);

#endif // defined(ASIO_HAS_STRING_VIEW)
       //  || defined(GENERATING_DOCUMENTATION)

#if !defined(ASIO_NO_IOSTREAM)

/// Output a network as a string.
/**
 * Used to output a human-readable string for a specified network.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param net The network to be written.
 *
 * @return The output stream.
 *
 * @relates asio::ip::address_v4
 */
template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const network_v4& net);

#endif // !defined(ASIO_NO_IOSTREAM)

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/ip/impl/network_v4.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/impl/network_v4.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_IP_NETWORK_V4_HPP
