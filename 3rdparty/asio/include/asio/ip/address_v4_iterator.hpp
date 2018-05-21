//
// ip/address_v4_iterator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_ADDRESS_V4_ITERATOR_HPP
#define ASIO_IP_ADDRESS_V4_ITERATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/ip/address_v4.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

template <typename> class basic_address_iterator;

/// An input iterator that can be used for traversing IPv4 addresses.
/**
 * In addition to satisfying the input iterator requirements, this iterator
 * also supports decrement.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <> class basic_address_iterator<address_v4>
{
public:
  /// The type of the elements pointed to by the iterator.
  typedef address_v4 value_type;

  /// Distance between two iterators.
  typedef std::ptrdiff_t difference_type;

  /// The type of a pointer to an element pointed to by the iterator.
  typedef const address_v4* pointer;

  /// The type of a reference to an element pointed to by the iterator.
  typedef const address_v4& reference;

  /// Denotes that the iterator satisfies the input iterator requirements.
  typedef std::input_iterator_tag iterator_category;

  /// Construct an iterator that points to the specified address.
  basic_address_iterator(const address_v4& addr) ASIO_NOEXCEPT
    : address_(addr)
  {
  }

  /// Copy constructor.
  basic_address_iterator(
      const basic_address_iterator& other) ASIO_NOEXCEPT
    : address_(other.address_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  /// Move constructor.
  basic_address_iterator(basic_address_iterator&& other) ASIO_NOEXCEPT
    : address_(ASIO_MOVE_CAST(address_v4)(other.address_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Assignment operator.
  basic_address_iterator& operator=(
      const basic_address_iterator& other) ASIO_NOEXCEPT
  {
    address_ = other.address_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE)
  /// Move assignment operator.
  basic_address_iterator& operator=(
      basic_address_iterator&& other) ASIO_NOEXCEPT
  {
    address_ = ASIO_MOVE_CAST(address_v4)(other.address_);
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Dereference the iterator.
  const address_v4& operator*() const ASIO_NOEXCEPT
  {
    return address_;
  }

  /// Dereference the iterator.
  const address_v4* operator->() const ASIO_NOEXCEPT
  {
    return &address_;
  }

  /// Pre-increment operator.
  basic_address_iterator& operator++() ASIO_NOEXCEPT
  {
    address_ = address_v4((address_.to_uint() + 1) & 0xFFFFFFFF);
    return *this;
  }

  /// Post-increment operator.
  basic_address_iterator operator++(int) ASIO_NOEXCEPT
  {
    basic_address_iterator tmp(*this);
    ++*this;
    return tmp;
  }

  /// Pre-decrement operator.
  basic_address_iterator& operator--() ASIO_NOEXCEPT
  {
    address_ = address_v4((address_.to_uint() - 1) & 0xFFFFFFFF);
    return *this;
  }

  /// Post-decrement operator.
  basic_address_iterator operator--(int)
  {
    basic_address_iterator tmp(*this);
    --*this;
    return tmp;
  }

  /// Compare two addresses for equality.
  friend bool operator==(const basic_address_iterator& a,
      const basic_address_iterator& b)
  {
    return a.address_ == b.address_;
  }

  /// Compare two addresses for inequality.
  friend bool operator!=(const basic_address_iterator& a,
      const basic_address_iterator& b)
  {
    return a.address_ != b.address_;
  }

private:
  address_v4 address_;
};

/// An input iterator that can be used for traversing IPv4 addresses.
typedef basic_address_iterator<address_v4> address_v4_iterator;

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_ADDRESS_V4_ITERATOR_HPP
