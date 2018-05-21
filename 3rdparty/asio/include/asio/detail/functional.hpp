//
// detail/functional.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_FUNCTIONAL_HPP
#define ASIO_DETAIL_FUNCTIONAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <functional>

#if !defined(ASIO_HAS_STD_FUNCTION)
# include <boost/function.hpp>
#endif // !defined(ASIO_HAS_STD_FUNCTION)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_FUNCTION)
using std::function;
#else // defined(ASIO_HAS_STD_FUNCTION)
using boost::function;
#endif // defined(ASIO_HAS_STD_FUNCTION)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_FUNCTIONAL_HPP
