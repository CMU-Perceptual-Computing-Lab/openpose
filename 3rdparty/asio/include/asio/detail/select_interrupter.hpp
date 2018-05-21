//
// detail/select_interrupter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
# include "asio/detail/socket_select_interrupter.hpp"
#elif defined(ASIO_HAS_EVENTFD)
# include "asio/detail/eventfd_select_interrupter.hpp"
#else
# include "asio/detail/pipe_select_interrupter.hpp"
#endif

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
typedef socket_select_interrupter select_interrupter;
#elif defined(ASIO_HAS_EVENTFD)
typedef eventfd_select_interrupter select_interrupter;
#else
typedef pipe_select_interrupter select_interrupter;
#endif

} // namespace detail
} // namespace asio

#endif // !defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_SELECT_INTERRUPTER_HPP
