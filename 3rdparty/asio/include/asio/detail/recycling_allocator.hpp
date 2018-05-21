//
// detail/recycling_allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_RECYCLING_ALLOCATOR_HPP
#define ASIO_DETAIL_RECYCLING_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/thread_context.hpp"
#include "asio/detail/thread_info_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class recycling_allocator
{
public:
  typedef T value_type;

  template <typename U>
  struct rebind
  {
    typedef recycling_allocator<U> other;
  };

  recycling_allocator()
  {
  }

  template <typename U>
  recycling_allocator(const recycling_allocator<U>&)
  {
  }

  T* allocate(std::size_t n)
  {
    typedef thread_context::thread_call_stack call_stack;
    void* p = thread_info_base::allocate(call_stack::top(), sizeof(T) * n);
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t n)
  {
    typedef thread_context::thread_call_stack call_stack;
    thread_info_base::deallocate(call_stack::top(), p, sizeof(T) * n);
  }
};

template <>
class recycling_allocator<void>
{
public:
  typedef void value_type;

  template <typename U>
  struct rebind
  {
    typedef recycling_allocator<U> other;
  };

  recycling_allocator()
  {
  }

  template <typename U>
  recycling_allocator(const recycling_allocator<U>&)
  {
  }
};

template <typename Allocator>
struct get_recycling_allocator
{
  typedef Allocator type;
  static type get(const Allocator& a) { return a; }
};

template <typename T>
struct get_recycling_allocator<std::allocator<T> >
{
  typedef recycling_allocator<T> type;
  static type get(const std::allocator<T>&) { return type(); }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_RECYCLING_ALLOCATOR_HPP
