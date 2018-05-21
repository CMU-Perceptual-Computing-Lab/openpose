//
// detail/reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTOR_HPP
#define ASIO_DETAIL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/reactor_fwd.hpp"

#if defined(ASIO_HAS_EPOLL)
# include "asio/detail/epoll_reactor.hpp"
#elif defined(ASIO_HAS_KQUEUE)
# include "asio/detail/kqueue_reactor.hpp"
#elif defined(ASIO_HAS_DEV_POLL)
# include "asio/detail/dev_poll_reactor.hpp"
#elif defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/null_reactor.hpp"
#else
# include "asio/detail/select_reactor.hpp"
#endif

#endif // ASIO_DETAIL_REACTOR_HPP
