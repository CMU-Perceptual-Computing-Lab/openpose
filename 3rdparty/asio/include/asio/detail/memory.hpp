//
// detail/memory.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_MEMORY_HPP
#define ASIO_DETAIL_MEMORY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <memory>

#if !defined(ASIO_HAS_STD_SHARED_PTR)
# include <boost/shared_ptr.hpp>
# include <boost/weak_ptr.hpp>
#endif // !defined(ASIO_HAS_STD_SHARED_PTR)

#if !defined(ASIO_HAS_STD_ADDRESSOF)
# include <boost/utility/addressof.hpp>
#endif // !defined(ASIO_HAS_STD_ADDRESSOF)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_SHARED_PTR)
using std::shared_ptr;
using std::weak_ptr;
#else // defined(ASIO_HAS_STD_SHARED_PTR)
using boost::shared_ptr;
using boost::weak_ptr;
#endif // defined(ASIO_HAS_STD_SHARED_PTR)

#if defined(ASIO_HAS_STD_ADDRESSOF)
using std::addressof;
#else // defined(ASIO_HAS_STD_ADDRESSOF)
using boost::addressof;
#endif // defined(ASIO_HAS_STD_ADDRESSOF)

} // namespace detail

#if defined(ASIO_HAS_CXX11_ALLOCATORS)
using std::allocator_arg_t;
# define ASIO_USES_ALLOCATOR(t) \
  namespace std { \
    template <typename Allocator> \
    struct uses_allocator<t, Allocator> : true_type {}; \
  } \
  /**/
# define ASIO_REBIND_ALLOC(alloc, t) \
  typename std::allocator_traits<alloc>::template rebind_alloc<t>
  /**/
#else // defined(ASIO_HAS_CXX11_ALLOCATORS)
struct allocator_arg_t {};
# define ASIO_USES_ALLOCATOR(t)
# define ASIO_REBIND_ALLOC(alloc, t) \
  typename alloc::template rebind<t>::other
  /**/
#endif // defined(ASIO_HAS_CXX11_ALLOCATORS)

} // namespace asio

#endif // ASIO_DETAIL_MEMORY_HPP
