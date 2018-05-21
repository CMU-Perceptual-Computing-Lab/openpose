//
// ip/detail/endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_DETAIL_ENDPOINT_HPP
#define ASIO_IP_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <string>
#include "asio/detail/socket_types.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/error_code.hpp"
#include "asio/ip/address.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {
namespace detail {

// Helper class for implementating an IP endpoint.
class endpoint
{
public:
  // Default constructor.
  ASIO_DECL endpoint();

  // Construct an endpoint using a family and port number.
  ASIO_DECL endpoint(int family, unsigned short port_num);

  // Construct an endpoint using an address and port number.
  ASIO_DECL endpoint(const asio::ip::address& addr,
      unsigned short port_num);

  // Copy constructor.
  endpoint(const endpoint& other)
    : data_(other.data_)
  {
  }

  // Assign from another endpoint.
  endpoint& operator=(const endpoint& other)
  {
    data_ = other.data_;
    return *this;
  }

  // Get the underlying endpoint in the native type.
  asio::detail::socket_addr_type* data()
  {
    return &data_.base;
  }

  // Get the underlying endpoint in the native type.
  const asio::detail::socket_addr_type* data() const
  {
    return &data_.base;
  }

  // Get the underlying size of the endpoint in the native type.
  std::size_t size() const
  {
    if (is_v4())
      return sizeof(asio::detail::sockaddr_in4_type);
    else
      return sizeof(asio::detail::sockaddr_in6_type);
  }

  // Set the underlying size of the endpoint in the native type.
  ASIO_DECL void resize(std::size_t new_size);

  // Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return sizeof(data_);
  }

  // Get the port associated with the endpoint.
  ASIO_DECL unsigned short port() const;

  // Set the port associated with the endpoint.
  ASIO_DECL void port(unsigned short port_num);

  // Get the IP address associated with the endpoint.
  ASIO_DECL asio::ip::address address() const;

  // Set the IP address associated with the endpoint.
  ASIO_DECL void address(const asio::ip::address& addr);

  // Compare two endpoints for equality.
  ASIO_DECL friend bool operator==(
      const endpoint& e1, const endpoint& e2);

  // Compare endpoints for ordering.
  ASIO_DECL friend bool operator<(
      const endpoint& e1, const endpoint& e2);

  // Determine whether the endpoint is IPv4.
  bool is_v4() const
  {
    return data_.base.sa_family == ASIO_OS_DEF(AF_INET);
  }

#if !defined(ASIO_NO_IOSTREAM)
  // Convert to a string.
  ASIO_DECL std::string to_string() const;
#endif // !defined(ASIO_NO_IOSTREAM)

private:
  // The underlying IP socket address.
  union data_union
  {
    asio::detail::socket_addr_type base;
    asio::detail::sockaddr_in4_type v4;
    asio::detail::sockaddr_in6_type v6;
  } data_;
};

} // namespace detail
} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/detail/impl/endpoint.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_IP_DETAIL_ENDPOINT_HPP
