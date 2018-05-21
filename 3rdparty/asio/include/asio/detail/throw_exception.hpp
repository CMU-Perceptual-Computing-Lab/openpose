//
// detail/throw_exception.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_THROW_EXCEPTION_HPP
#define ASIO_DETAIL_THROW_EXCEPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_THROW_EXCEPTION)
# include <boost/throw_exception.hpp>
#endif // defined(ASIO_BOOST_THROW_EXCEPTION)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_BOOST_THROW_EXCEPTION)
using boost::throw_exception;
#else // defined(ASIO_HAS_BOOST_THROW_EXCEPTION)

// Declare the throw_exception function for all targets.
template <typename Exception>
void throw_exception(const Exception& e);

// Only define the throw_exception function when exceptions are enabled.
// Otherwise, it is up to the application to provide a definition of this
// function.
# if !defined(ASIO_NO_EXCEPTIONS)
template <typename Exception>
void throw_exception(const Exception& e)
{
  throw e;
}
# endif // !defined(ASIO_NO_EXCEPTIONS)

#endif // defined(ASIO_HAS_BOOST_THROW_EXCEPTION)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_THROW_EXCEPTION_HPP
