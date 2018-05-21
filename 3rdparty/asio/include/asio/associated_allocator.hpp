//
// associated_allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ASSOCIATED_ALLOCATOR_HPP
#define ASIO_ASSOCIATED_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <memory>
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename>
struct associated_allocator_check
{
  typedef void type;
};

template <typename T, typename E, typename = void>
struct associated_allocator_impl
{
  typedef E type;

  static type get(const T&, const E& e) ASIO_NOEXCEPT
  {
    return e;
  }
};

template <typename T, typename E>
struct associated_allocator_impl<T, E,
  typename associated_allocator_check<typename T::allocator_type>::type>
{
  typedef typename T::allocator_type type;

  static type get(const T& t, const E&) ASIO_NOEXCEPT
  {
    return t.get_allocator();
  }
};

} // namespace detail

/// Traits type used to obtain the allocator associated with an object.
/**
 * A program may specialise this traits type if the @c T template parameter in
 * the specialisation is a user-defined type. The template parameter @c
 * Allocator shall be a type meeting the Allocator requirements.
 *
 * Specialisations shall meet the following requirements, where @c t is a const
 * reference to an object of type @c T, and @c a is an object of type @c
 * Allocator.
 *
 * @li Provide a nested typedef @c type that identifies a type meeting the
 * Allocator requirements.
 *
 * @li Provide a noexcept static member function named @c get, callable as @c
 * get(t) and with return type @c type.
 *
 * @li Provide a noexcept static member function named @c get, callable as @c
 * get(t,a) and with return type @c type.
 */
template <typename T, typename Allocator = std::allocator<void> >
struct associated_allocator
{
  /// If @c T has a nested type @c allocator_type, <tt>T::allocator_type</tt>.
  /// Otherwise @c Allocator.
#if defined(GENERATING_DOCUMENTATION)
  typedef see_below type;
#else // defined(GENERATING_DOCUMENTATION)
  typedef typename detail::associated_allocator_impl<T, Allocator>::type type;
#endif // defined(GENERATING_DOCUMENTATION)

  /// If @c T has a nested type @c allocator_type, returns
  /// <tt>t.get_allocator()</tt>. Otherwise returns @c a.
  static type get(const T& t,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return detail::associated_allocator_impl<T, Allocator>::get(t, a);
  }
};

/// Helper function to obtain an object's associated allocator.
/**
 * @returns <tt>associated_allocator<T>::get(t)</tt>
 */
template <typename T>
inline typename associated_allocator<T>::type
get_associated_allocator(const T& t) ASIO_NOEXCEPT
{
  return associated_allocator<T>::get(t);
}

/// Helper function to obtain an object's associated allocator.
/**
 * @returns <tt>associated_allocator<T, Allocator>::get(t, a)</tt>
 */
template <typename T, typename Allocator>
inline typename associated_allocator<T, Allocator>::type
get_associated_allocator(const T& t, const Allocator& a) ASIO_NOEXCEPT
{
  return associated_allocator<T, Allocator>::get(t, a);
}

#if defined(ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename Allocator = std::allocator<void> >
using associated_allocator_t
  = typename associated_allocator<T, Allocator>::type;

#endif // defined(ASIO_HAS_ALIAS_TEMPLATES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ASSOCIATED_ALLOCATOR_HPP
