//
// ip/address_v4.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_ADDRESS_V4_HPP
#define ASIO_IP_ADDRESS_V4_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <string>
#include "asio/detail/array.hpp"
#include "asio/detail/cstdint.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/string_view.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/error_code.hpp"

#if !defined(ASIO_NO_IOSTREAM)
# include <iosfwd>
#endif // !defined(ASIO_NO_IOSTREAM)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Implements IP version 4 style addresses.
/**
 * The asio::ip::address_v4 class provides the ability to use and
 * manipulate IP version 4 addresses.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class address_v4
{
public:
  /// The type used to represent an address as an unsigned integer.
  typedef uint_least32_t uint_type;

  /// The type used to represent an address as an array of bytes.
  /**
   * @note This type is defined in terms of the C++0x template @c std::array
   * when it is available. Otherwise, it uses @c boost:array.
   */
#if defined(GENERATING_DOCUMENTATION)
  typedef array<unsigned char, 4> bytes_type;
#else
  typedef asio::detail::array<unsigned char, 4> bytes_type;
#endif

  /// Default constructor.
  address_v4()
  {
    addr_.s_addr = 0;
  }

  /// Construct an address from raw bytes.
  ASIO_DECL explicit address_v4(const bytes_type& bytes);

  /// Construct an address from an unsigned integer in host byte order.
  ASIO_DECL explicit address_v4(uint_type addr);

  /// Copy constructor.
  address_v4(const address_v4& other)
    : addr_(other.addr_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  /// Move constructor.
  address_v4(address_v4&& other)
    : addr_(other.addr_)
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Assign from another address.
  address_v4& operator=(const address_v4& other)
  {
    addr_ = other.addr_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE)
  /// Move-assign from another address.
  address_v4& operator=(address_v4&& other)
  {
    addr_ = other.addr_;
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Get the address in bytes, in network byte order.
  ASIO_DECL bytes_type to_bytes() const;

  /// Get the address as an unsigned integer in host byte order
  ASIO_DECL uint_type to_uint() const;

#if !defined(ASIO_NO_DEPRECATED)
  /// Get the address as an unsigned long in host byte order
  ASIO_DECL unsigned long to_ulong() const;
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Get the address as a string in dotted decimal format.
  ASIO_DECL std::string to_string() const;

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use other overload.) Get the address as a string in dotted
  /// decimal format.
  ASIO_DECL std::string to_string(asio::error_code& ec) const;

  /// (Deprecated: Use make_address_v4().) Create an address from an IP address
  /// string in dotted decimal form.
  static address_v4 from_string(const char* str);

  /// (Deprecated: Use make_address_v4().) Create an address from an IP address
  /// string in dotted decimal form.
  static address_v4 from_string(
      const char* str, asio::error_code& ec);

  /// (Deprecated: Use make_address_v4().) Create an address from an IP address
  /// string in dotted decimal form.
  static address_v4 from_string(const std::string& str);

  /// (Deprecated: Use make_address_v4().) Create an address from an IP address
  /// string in dotted decimal form.
  static address_v4 from_string(
      const std::string& str, asio::error_code& ec);
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Determine whether the address is a loopback address.
  ASIO_DECL bool is_loopback() const;

  /// Determine whether the address is unspecified.
  ASIO_DECL bool is_unspecified() const;

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use network_v4 class.) Determine whether the address is a
  /// class A address.
  ASIO_DECL bool is_class_a() const;

  /// (Deprecated: Use network_v4 class.) Determine whether the address is a
  /// class B address.
  ASIO_DECL bool is_class_b() const;

  /// (Deprecated: Use network_v4 class.) Determine whether the address is a
  /// class C address.
  ASIO_DECL bool is_class_c() const;
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Determine whether the address is a multicast address.
  ASIO_DECL bool is_multicast() const;

  /// Compare two addresses for equality.
  friend bool operator==(const address_v4& a1, const address_v4& a2)
  {
    return a1.addr_.s_addr == a2.addr_.s_addr;
  }

  /// Compare two addresses for inequality.
  friend bool operator!=(const address_v4& a1, const address_v4& a2)
  {
    return a1.addr_.s_addr != a2.addr_.s_addr;
  }

  /// Compare addresses for ordering.
  friend bool operator<(const address_v4& a1, const address_v4& a2)
  {
    return a1.to_uint() < a2.to_uint();
  }

  /// Compare addresses for ordering.
  friend bool operator>(const address_v4& a1, const address_v4& a2)
  {
    return a1.to_uint() > a2.to_uint();
  }

  /// Compare addresses for ordering.
  friend bool operator<=(const address_v4& a1, const address_v4& a2)
  {
    return a1.to_uint() <= a2.to_uint();
  }

  /// Compare addresses for ordering.
  friend bool operator>=(const address_v4& a1, const address_v4& a2)
  {
    return a1.to_uint() >= a2.to_uint();
  }

  /// Obtain an address object that represents any address.
  static address_v4 any()
  {
    return address_v4();
  }

  /// Obtain an address object that represents the loopback address.
  static address_v4 loopback()
  {
    return address_v4(0x7F000001);
  }

  /// Obtain an address object that represents the broadcast address.
  static address_v4 broadcast()
  {
    return address_v4(0xFFFFFFFF);
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use network_v4 class.) Obtain an address object that
  /// represents the broadcast address that corresponds to the specified
  /// address and netmask.
  ASIO_DECL static address_v4 broadcast(
      const address_v4& addr, const address_v4& mask);

  /// (Deprecated: Use network_v4 class.) Obtain the netmask that corresponds
  /// to the address, based on its address class.
  ASIO_DECL static address_v4 netmask(const address_v4& addr);
#endif // !defined(ASIO_NO_DEPRECATED)

private:
  // The underlying IPv4 address.
  asio::detail::in4_addr_type addr_;
};

/// Create an IPv4 address from raw bytes in network order.
/**
 * @relates address_v4
 */
inline address_v4 make_address_v4(const address_v4::bytes_type& bytes)
{
  return address_v4(bytes);
}

/// Create an IPv4 address from an unsigned integer in host byte order.
/**
 * @relates address_v4
 */
inline address_v4 make_address_v4(address_v4::uint_type addr)
{
  return address_v4(addr);
}

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(const char* str);

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(
    const char* str, asio::error_code& ec);

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(const std::string& str);

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(
    const std::string& str, asio::error_code& ec);

#if defined(ASIO_HAS_STRING_VIEW) \
  || defined(GENERATING_DOCUMENTATION)

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(string_view str);

/// Create an IPv4 address from an IP address string in dotted decimal form.
/**
 * @relates address_v4
 */
ASIO_DECL address_v4 make_address_v4(
    string_view str, asio::error_code& ec);

#endif // defined(ASIO_HAS_STRING_VIEW)
       //  || defined(GENERATING_DOCUMENTATION)

#if !defined(ASIO_NO_IOSTREAM)

/// Output an address as a string.
/**
 * Used to output a human-readable string for a specified address.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param addr The address to be written.
 *
 * @return The output stream.
 *
 * @relates asio::ip::address_v4
 */
template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const address_v4& addr);

#endif // !defined(ASIO_NO_IOSTREAM)

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/ip/impl/address_v4.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/impl/address_v4.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_IP_ADDRESS_V4_HPP
