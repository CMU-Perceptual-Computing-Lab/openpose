//
// detail/array.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ARRAY_HPP
#define ASIO_DETAIL_ARRAY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_ARRAY)
# include <array>
#else // defined(ASIO_HAS_STD_ARRAY)
# include <boost/array.hpp>
#endif // defined(ASIO_HAS_STD_ARRAY)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_ARRAY)
using std::array;
#else // defined(ASIO_HAS_STD_ARRAY)
using boost::array;
#endif // defined(ASIO_HAS_STD_ARRAY)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_ARRAY_HPP
