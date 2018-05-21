//
// generic/detail/endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GENERIC_DETAIL_ENDPOINT_HPP
#define ASIO_GENERIC_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <cstddef>
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace generic {
namespace detail {

// Helper class for implementing a generic socket endpoint.
class endpoint
{
public:
  // Default constructor.
  ASIO_DECL endpoint();

  // Construct an endpoint from the specified raw bytes.
  ASIO_DECL endpoint(const void* sock_addr,
      std::size_t sock_addr_size, int sock_protocol);

  // Copy constructor.
  endpoint(const endpoint& other)
    : data_(other.data_),
      size_(other.size_),
      protocol_(other.protocol_)
  {
  }

  // Assign from another endpoint.
  endpoint& operator=(const endpoint& other)
  {
    data_ = other.data_;
    size_ = other.size_;
    protocol_ = other.protocol_;
    return *this;
  }

  // Get the address family associated with the endpoint.
  int family() const
  {
    return data_.base.sa_family;
  }

  // Get the socket protocol associated with the endpoint.
  int protocol() const
  {
    return protocol_;
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
    return size_;
  }

  // Set the underlying size of the endpoint in the native type.
  ASIO_DECL void resize(std::size_t size);

  // Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return sizeof(asio::detail::sockaddr_storage_type);
  }

  // Compare two endpoints for equality.
  ASIO_DECL friend bool operator==(
      const endpoint& e1, const endpoint& e2);

  // Compare endpoints for ordering.
  ASIO_DECL friend bool operator<(
      const endpoint& e1, const endpoint& e2);

private:
  // The underlying socket address.
  union data_union
  {
    asio::detail::socket_addr_type base;
    asio::detail::sockaddr_storage_type generic;
  } data_;

  // The length of the socket address stored in the endpoint.
  std::size_t size_;

  // The socket protocol associated with the endpoint.
  int protocol_;

  // Initialise with a specified memory.
  ASIO_DECL void init(const void* sock_addr,
      std::size_t sock_addr_size, int sock_protocol);
};

} // namespace detail
} // namespace generic
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/generic/detail/impl/endpoint.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_GENERIC_DETAIL_ENDPOINT_HPP
