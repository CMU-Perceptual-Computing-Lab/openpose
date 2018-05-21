//
// detail/work_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WORK_DISPATCHER_HPP
#define ASIO_DETAIL_WORK_DISPATCHER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_executor.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/executor_work_guard.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler>
class work_dispatcher
{
public:
  work_dispatcher(Handler& handler)
    : work_((get_associated_executor)(handler)),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
  }

#if defined(ASIO_HAS_MOVE)
  work_dispatcher(const work_dispatcher& other)
    : work_(other.work_),
      handler_(other.handler_)
  {
  }

  work_dispatcher(work_dispatcher&& other)
    : work_(ASIO_MOVE_CAST(executor_work_guard<
        typename associated_executor<Handler>::type>)(other.work_)),
      handler_(ASIO_MOVE_CAST(Handler)(other.handler_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  void operator()()
  {
    typename associated_allocator<Handler>::type alloc(
        (get_associated_allocator)(handler_));
    work_.get_executor().dispatch(
        ASIO_MOVE_CAST(Handler)(handler_), alloc);
    work_.reset();
  }

private:
  executor_work_guard<typename associated_executor<Handler>::type> work_;
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WORK_DISPATCHER_HPP
