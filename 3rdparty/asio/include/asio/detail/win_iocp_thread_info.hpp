//
// detail/win_iocp_thread_info.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_THREAD_INFO_HPP
#define ASIO_DETAIL_WIN_IOCP_THREAD_INFO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/thread_info_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct win_iocp_thread_info : public thread_info_base
{
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_THREAD_INFO_HPP
