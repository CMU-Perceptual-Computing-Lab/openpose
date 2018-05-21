//
// detail/timer_scheduler_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TIMER_SCHEDULER_FWD_HPP
#define ASIO_DETAIL_TIMER_SCHEDULER_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS_RUNTIME)
typedef class winrt_timer_scheduler timer_scheduler;
#elif defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context timer_scheduler;
#elif defined(ASIO_HAS_EPOLL)
typedef class epoll_reactor timer_scheduler;
#elif defined(ASIO_HAS_KQUEUE)
typedef class kqueue_reactor timer_scheduler;
#elif defined(ASIO_HAS_DEV_POLL)
typedef class dev_poll_reactor timer_scheduler;
#else
typedef class select_reactor timer_scheduler;
#endif

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_TIMER_SCHEDULER_FWD_HPP
