//
// detail/chrono.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CHRONO_HPP
#define ASIO_DETAIL_CHRONO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_CHRONO)
# include <chrono>
#elif defined(ASIO_HAS_BOOST_CHRONO)
# include <boost/chrono/system_clocks.hpp>
#endif // defined(ASIO_HAS_BOOST_CHRONO)

namespace asio {
namespace chrono {

#if defined(ASIO_HAS_STD_CHRONO)
using std::chrono::duration;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;
using std::chrono::time_point_cast;
#if defined(ASIO_HAS_STD_CHRONO_MONOTONIC_CLOCK)
typedef std::chrono::monotonic_clock steady_clock;
#else // defined(ASIO_HAS_STD_CHRONO_MONOTONIC_CLOCK)
using std::chrono::steady_clock;
#endif // defined(ASIO_HAS_STD_CHRONO_MONOTONIC_CLOCK)
using std::chrono::system_clock;
using std::chrono::high_resolution_clock;
#elif defined(ASIO_HAS_BOOST_CHRONO)
using boost::chrono::duration;
using boost::chrono::time_point;
using boost::chrono::duration_cast;
using boost::chrono::nanoseconds;
using boost::chrono::microseconds;
using boost::chrono::milliseconds;
using boost::chrono::seconds;
using boost::chrono::minutes;
using boost::chrono::hours;
using boost::chrono::time_point_cast;
using boost::chrono::system_clock;
using boost::chrono::steady_clock;
using boost::chrono::high_resolution_clock;
#endif // defined(ASIO_HAS_BOOST_CHRONO)

} // namespace chrono
} // namespace asio

#endif // ASIO_DETAIL_CHRONO_HPP
