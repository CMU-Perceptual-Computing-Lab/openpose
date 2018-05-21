//
// ip/basic_resolver_iterator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_BASIC_RESOLVER_ITERATOR_HPP
#define ASIO_IP_BASIC_RESOLVER_ITERATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstddef>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>
#include "asio/detail/memory.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ip/basic_resolver_entry.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/winrt_utils.hpp"
#endif // defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// An iterator over the entries produced by a resolver.
/**
 * The asio::ip::basic_resolver_iterator class template is used to define
 * iterators over the results returned by a resolver.
 *
 * The iterator's value_type, obtained when the iterator is dereferenced, is:
 * @code const basic_resolver_entry<InternetProtocol> @endcode
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <typename InternetProtocol>
class basic_resolver_iterator
{
public:
  /// The type used for the distance between two iterators.
  typedef std::ptrdiff_t difference_type;

  /// The type of the value pointed to by the iterator.
  typedef basic_resolver_entry<InternetProtocol> value_type;

  /// The type of the result of applying operator->() to the iterator.
  typedef const basic_resolver_entry<InternetProtocol>* pointer;

  /// The type of the result of applying operator*() to the iterator.
  typedef const basic_resolver_entry<InternetProtocol>& reference;

  /// The iterator category.
  typedef std::forward_iterator_tag iterator_category;

  /// Default constructor creates an end iterator.
  basic_resolver_iterator()
    : index_(0)
  {
  }

  /// Copy constructor.
  basic_resolver_iterator(const basic_resolver_iterator& other)
    : values_(other.values_),
      index_(other.index_)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move constructor.
  basic_resolver_iterator(basic_resolver_iterator&& other)
    : values_(ASIO_MOVE_CAST(values_ptr_type)(other.values_)),
      index_(other.index_)
  {
    other.index_ = 0;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Assignment operator.
  basic_resolver_iterator& operator=(const basic_resolver_iterator& other)
  {
    values_ = other.values_;
    index_ = other.index_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-assignment operator.
  basic_resolver_iterator& operator=(basic_resolver_iterator&& other)
  {
    if (this != &other)
    {
      values_ = ASIO_MOVE_CAST(values_ptr_type)(other.values_);
      index_ = other.index_;
      other.index_ = 0;
    }

    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Dereference an iterator.
  const basic_resolver_entry<InternetProtocol>& operator*() const
  {
    return dereference();
  }

  /// Dereference an iterator.
  const basic_resolver_entry<InternetProtocol>* operator->() const
  {
    return &dereference();
  }

  /// Increment operator (prefix).
  basic_resolver_iterator& operator++()
  {
    increment();
    return *this;
  }

  /// Increment operator (postfix).
  basic_resolver_iterator operator++(int)
  {
    basic_resolver_iterator tmp(*this);
    ++*this;
    return tmp;
  }

  /// Test two iterators for equality.
  friend bool operator==(const basic_resolver_iterator& a,
      const basic_resolver_iterator& b)
  {
    return a.equal(b);
  }

  /// Test two iterators for inequality.
  friend bool operator!=(const basic_resolver_iterator& a,
      const basic_resolver_iterator& b)
  {
    return !a.equal(b);
  }

protected:
  void increment()
  {
    if (++index_ == values_->size())
    {
      // Reset state to match a default constructed end iterator.
      values_.reset();
      index_ = 0;
    }
  }

  bool equal(const basic_resolver_iterator& other) const
  {
    if (!values_ && !other.values_)
      return true;
    if (values_ != other.values_)
      return false;
    return index_ == other.index_;
  }

  const basic_resolver_entry<InternetProtocol>& dereference() const
  {
    return (*values_)[index_];
  }

  typedef std::vector<basic_resolver_entry<InternetProtocol> > values_type;
  typedef asio::detail::shared_ptr<values_type> values_ptr_type;
  values_ptr_type values_;
  std::size_t index_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_BASIC_RESOLVER_ITERATOR_HPP
