//
// detail/scheduler_thread_info.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP
#define ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/op_queue.hpp"
#include "asio/detail/thread_info_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class scheduler;
class scheduler_operation;

struct scheduler_thread_info : public thread_info_base
{
  op_queue<scheduler_operation> private_op_queue;
  long private_outstanding_work;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP
