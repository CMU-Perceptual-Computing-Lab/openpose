//
// detail/limits.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_LIMITS_HPP
#define ASIO_DETAIL_LIMITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_LIMITS)
# include <boost/limits.hpp>
#else // defined(ASIO_HAS_BOOST_LIMITS)
# include <limits>
#endif // defined(ASIO_HAS_BOOST_LIMITS)

#endif // ASIO_DETAIL_LIMITS_HPP
