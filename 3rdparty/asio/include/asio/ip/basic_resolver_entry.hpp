//
// ip/basic_resolver_entry.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_BASIC_RESOLVER_ENTRY_HPP
#define ASIO_IP_BASIC_RESOLVER_ENTRY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <string>
#include "asio/detail/string_view.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// An entry produced by a resolver.
/**
 * The asio::ip::basic_resolver_entry class template describes an entry
 * as returned by a resolver.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <typename InternetProtocol>
class basic_resolver_entry
{
public:
  /// The protocol type associated with the endpoint entry.
  typedef InternetProtocol protocol_type;

  /// The endpoint type associated with the endpoint entry.
  typedef typename InternetProtocol::endpoint endpoint_type;

  /// Default constructor.
  basic_resolver_entry()
  {
  }

  /// Construct with specified endpoint, host name and service name.
  basic_resolver_entry(const endpoint_type& ep,
      ASIO_STRING_VIEW_PARAM host, ASIO_STRING_VIEW_PARAM service)
    : endpoint_(ep),
      host_name_(static_cast<std::string>(host)),
      service_name_(static_cast<std::string>(service))
  {
  }

  /// Get the endpoint associated with the entry.
  endpoint_type endpoint() const
  {
    return endpoint_;
  }

  /// Convert to the endpoint associated with the entry.
  operator endpoint_type() const
  {
    return endpoint_;
  }

  /// Get the host name associated with the entry.
  std::string host_name() const
  {
    return host_name_;
  }

  /// Get the host name associated with the entry.
  template <class Allocator>
  std::basic_string<char, std::char_traits<char>, Allocator> host_name(
      const Allocator& alloc = Allocator()) const
  {
    return std::basic_string<char, std::char_traits<char>, Allocator>(
        host_name_.c_str(), alloc);
  }

  /// Get the service name associated with the entry.
  std::string service_name() const
  {
    return service_name_;
  }

  /// Get the service name associated with the entry.
  template <class Allocator>
  std::basic_string<char, std::char_traits<char>, Allocator> service_name(
      const Allocator& alloc = Allocator()) const
  {
    return std::basic_string<char, std::char_traits<char>, Allocator>(
        service_name_.c_str(), alloc);
  }

private:
  endpoint_type endpoint_;
  std::string host_name_;
  std::string service_name_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_BASIC_RESOLVER_ENTRY_HPP
