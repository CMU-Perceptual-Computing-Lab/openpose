//
// detail/win_iocp_overlapped_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_OVERLAPPED_OP_HPP
#define ASIO_DETAIL_WIN_IOCP_OVERLAPPED_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler>
class win_iocp_overlapped_op : public operation
{
public:
  ASIO_DEFINE_HANDLER_PTR(win_iocp_overlapped_op);

  win_iocp_overlapped_op(Handler& handler)
    : operation(&win_iocp_overlapped_op::do_complete),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
    handler_work<Handler>::start(handler_);
  }

  static void do_complete(void* owner, operation* base,
      const asio::error_code& ec, std::size_t bytes_transferred)
  {
    // Take ownership of the operation object.
    win_iocp_overlapped_op* o(static_cast<win_iocp_overlapped_op*>(base));
    ptr p = { asio::detail::addressof(o->handler_), o, o };
    handler_work<Handler> w(o->handler_);

    ASIO_HANDLER_COMPLETION((*o));

    // Make a copy of the handler so that the memory can be deallocated before
    // the upcall is made. Even if we're not about to make an upcall, a
    // sub-object of the handler may be the true owner of the memory associated
    // with the handler. Consequently, a local copy of the handler is required
    // to ensure that any owning sub-object remains valid until after we have
    // deallocated the memory here.
    detail::binder2<Handler, asio::error_code, std::size_t>
      handler(o->handler_, ec, bytes_transferred);
    p.h = asio::detail::addressof(handler.handler_);
    p.reset();

    // Make the upcall if required.
    if (owner)
    {
      fenced_block b(fenced_block::half);
      ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
      w.complete(handler, handler.handler_);
      ASIO_HANDLER_INVOCATION_END;
    }
  }

private:
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_OVERLAPPED_OP_HPP
