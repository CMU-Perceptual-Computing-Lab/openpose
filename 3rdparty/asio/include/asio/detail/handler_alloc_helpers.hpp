//
// detail/handler_alloc_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/recycling_allocator.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/handler_alloc_hook.hpp"

#include "asio/detail/push_options.hpp"

// Calls to asio_handler_allocate and asio_handler_deallocate must be made from
// a namespace that does not contain any overloads of these functions. The
// asio_handler_alloc_helpers namespace is defined here for that purpose.
namespace asio_handler_alloc_helpers {

template <typename Handler>
inline void* allocate(std::size_t s, Handler& h)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
  return ::operator new(s);
#else
  using asio::asio_handler_allocate;
  return asio_handler_allocate(s, asio::detail::addressof(h));
#endif
}

template <typename Handler>
inline void deallocate(void* p, std::size_t s, Handler& h)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
  ::operator delete(p);
#else
  using asio::asio_handler_deallocate;
  asio_handler_deallocate(p, s, asio::detail::addressof(h));
#endif
}

} // namespace asio_handler_alloc_helpers

namespace asio {
namespace detail {

template <typename Handler, typename T>
class hook_allocator
{
public:
  typedef T value_type;

  template <typename U>
  struct rebind
  {
    typedef hook_allocator<Handler, U> other;
  };

  explicit hook_allocator(Handler& h)
    : handler_(h)
  {
  }

  template <typename U>
  hook_allocator(const hook_allocator<Handler, U>& a)
    : handler_(a.handler_)
  {
  }

  T* allocate(std::size_t n)
  {
    return static_cast<T*>(
        asio_handler_alloc_helpers::allocate(sizeof(T) * n, handler_));
  }

  void deallocate(T* p, std::size_t n)
  {
    asio_handler_alloc_helpers::deallocate(p, sizeof(T) * n, handler_);
  }

//private:
  Handler& handler_;
};

template <typename Handler>
class hook_allocator<Handler, void>
{
public:
  typedef void value_type;

  template <typename U>
  struct rebind
  {
    typedef hook_allocator<Handler, U> other;
  };

  explicit hook_allocator(Handler& h)
    : handler_(h)
  {
  }

  template <typename U>
  hook_allocator(const hook_allocator<Handler, U>& a)
    : handler_(a.handler_)
  {
  }

//private:
  Handler& handler_;
};

template <typename Handler, typename Allocator>
struct get_hook_allocator
{
  typedef Allocator type;

  static type get(Handler&, const Allocator& a)
  {
    return a;
  }
};

template <typename Handler, typename T>
struct get_hook_allocator<Handler, std::allocator<T> >
{
  typedef hook_allocator<Handler, T> type;

  static type get(Handler& handler, const std::allocator<T>&)
  {
    return type(handler);
  }
};

} // namespace detail
} // namespace asio

#define ASIO_DEFINE_HANDLER_PTR(op) \
  struct ptr \
  { \
    Handler* h; \
    op* v; \
    op* p; \
    ~ptr() \
    { \
      reset(); \
    } \
    static op* allocate(Handler& handler) \
    { \
      typedef typename ::asio::associated_allocator< \
        Handler>::type associated_allocator_type; \
      typedef typename ::asio::detail::get_hook_allocator< \
        Handler, associated_allocator_type>::type hook_allocator_type; \
      ASIO_REBIND_ALLOC(hook_allocator_type, op) a( \
            ::asio::detail::get_hook_allocator< \
              Handler, associated_allocator_type>::get( \
                handler, ::asio::get_associated_allocator(handler))); \
      return a.allocate(1); \
    } \
    void reset() \
    { \
      if (p) \
      { \
        p->~op(); \
        p = 0; \
      } \
      if (v) \
      { \
        typedef typename ::asio::associated_allocator< \
          Handler>::type associated_allocator_type; \
        typedef typename ::asio::detail::get_hook_allocator< \
          Handler, associated_allocator_type>::type hook_allocator_type; \
        ASIO_REBIND_ALLOC(hook_allocator_type, op) a( \
              ::asio::detail::get_hook_allocator< \
                Handler, associated_allocator_type>::get( \
                  *h, ::asio::get_associated_allocator(*h))); \
        a.deallocate(static_cast<op*>(v), 1); \
        v = 0; \
      } \
    } \
  } \
  /**/

#define ASIO_DEFINE_HANDLER_ALLOCATOR_PTR(op) \
  struct ptr \
  { \
    const Alloc* a; \
    void* v; \
    op* p; \
    ~ptr() \
    { \
      reset(); \
    } \
    static op* allocate(const Alloc& a) \
    { \
      typedef typename ::asio::detail::get_recycling_allocator< \
        Alloc>::type recycling_allocator_type; \
      ASIO_REBIND_ALLOC(recycling_allocator_type, op) a1( \
            ::asio::detail::get_recycling_allocator<Alloc>::get(a)); \
      return a1.allocate(1); \
    } \
    void reset() \
    { \
      if (p) \
      { \
        p->~op(); \
        p = 0; \
      } \
      if (v) \
      { \
        typedef typename ::asio::detail::get_recycling_allocator< \
          Alloc>::type recycling_allocator_type; \
        ASIO_REBIND_ALLOC(recycling_allocator_type, op) a1( \
              ::asio::detail::get_recycling_allocator<Alloc>::get(*a)); \
        a1.deallocate(static_cast<op*>(v), 1); \
        v = 0; \
      } \
    } \
  } \
  /**/

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
