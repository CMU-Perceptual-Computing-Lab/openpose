//
// detail/atomic_count.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ATOMIC_COUNT_HPP
#define ASIO_DETAIL_ATOMIC_COUNT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)
// Nothing to include.
#elif defined(ASIO_HAS_STD_ATOMIC)
# include <atomic>
#else // defined(ASIO_HAS_STD_ATOMIC)
# include <boost/detail/atomic_count.hpp>
#endif // defined(ASIO_HAS_STD_ATOMIC)

namespace asio {
namespace detail {

#if !defined(ASIO_HAS_THREADS)
typedef long atomic_count;
inline void increment(atomic_count& a, long b) { a += b; }
#elif defined(ASIO_HAS_STD_ATOMIC)
typedef std::atomic<long> atomic_count;
inline void increment(atomic_count& a, long b) { a += b; }
#else // defined(ASIO_HAS_STD_ATOMIC)
typedef boost::detail::atomic_count atomic_count;
inline void increment(atomic_count& a, long b) { while (b > 0) ++a, --b; }
#endif // defined(ASIO_HAS_STD_ATOMIC)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_ATOMIC_COUNT_HPP
