//
// local/detail/endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Derived from a public domain implementation written by Daniel Casimiro.
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCAL_DETAIL_ENDPOINT_HPP
#define ASIO_LOCAL_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS)

#include <cstddef>
#include <string>
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {
namespace detail {

// Helper class for implementing a UNIX domain endpoint.
class endpoint
{
public:
  // Default constructor.
  ASIO_DECL endpoint();

  // Construct an endpoint using the specified path name.
  ASIO_DECL endpoint(const char* path_name);

  // Construct an endpoint using the specified path name.
  ASIO_DECL endpoint(const std::string& path_name);

  // Copy constructor.
  endpoint(const endpoint& other)
    : data_(other.data_),
      path_length_(other.path_length_)
  {
  }

  // Assign from another endpoint.
  endpoint& operator=(const endpoint& other)
  {
    data_ = other.data_;
    path_length_ = other.path_length_;
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
    return path_length_
      + offsetof(asio::detail::sockaddr_un_type, sun_path);
  }

  // Set the underlying size of the endpoint in the native type.
  ASIO_DECL void resize(std::size_t size);

  // Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return sizeof(asio::detail::sockaddr_un_type);
  }

  // Get the path associated with the endpoint.
  ASIO_DECL std::string path() const;

  // Set the path associated with the endpoint.
  ASIO_DECL void path(const char* p);

  // Set the path associated with the endpoint.
  ASIO_DECL void path(const std::string& p);

  // Compare two endpoints for equality.
  ASIO_DECL friend bool operator==(
      const endpoint& e1, const endpoint& e2);

  // Compare endpoints for ordering.
  ASIO_DECL friend bool operator<(
      const endpoint& e1, const endpoint& e2);

private:
  // The underlying UNIX socket address.
  union data_union
  {
    asio::detail::socket_addr_type base;
    asio::detail::sockaddr_un_type local;
  } data_;

  // The length of the path associated with the endpoint.
  std::size_t path_length_;

  // Initialise with a specified path.
  ASIO_DECL void init(const char* path, std::size_t path_length);
};

} // namespace detail
} // namespace local
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/local/detail/impl/endpoint.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_LOCAL_SOCKETS)

#endif // ASIO_LOCAL_DETAIL_ENDPOINT_HPP
