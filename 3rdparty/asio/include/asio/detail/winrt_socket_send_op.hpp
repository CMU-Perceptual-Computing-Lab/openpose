//
// detail/winrt_socket_send_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINRT_SOCKET_SEND_OP_HPP
#define ASIO_DETAIL_WINRT_SOCKET_SEND_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/winrt_async_op.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename ConstBufferSequence, typename Handler>
class winrt_socket_send_op :
  public winrt_async_op<unsigned int>
{
public:
  ASIO_DEFINE_HANDLER_PTR(winrt_socket_send_op);

  winrt_socket_send_op(const ConstBufferSequence& buffers, Handler& handler)
    : winrt_async_op<unsigned int>(&winrt_socket_send_op::do_complete),
      buffers_(buffers),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
    handler_work<Handler>::start(handler_);
  }

  static void do_complete(void* owner, operation* base,
      const asio::error_code&, std::size_t)
  {
    // Take ownership of the operation object.
    winrt_socket_send_op* o(static_cast<winrt_socket_send_op*>(base));
    ptr p = { asio::detail::addressof(o->handler_), o, o };
    handler_work<Handler> w(o->handler_);

    ASIO_HANDLER_COMPLETION((*o));

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
    // Check whether buffers are still valid.
    if (owner)
    {
      buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::validate(o->buffers_);
    }
#endif // defined(ASIO_ENABLE_BUFFER_DEBUGGING)

    // Make a copy of the handler so that the memory can be deallocated before
    // the upcall is made. Even if we're not about to make an upcall, a
    // sub-object of the handler may be the true owner of the memory associated
    // with the handler. Consequently, a local copy of the handler is required
    // to ensure that any owning sub-object remains valid until after we have
    // deallocated the memory here.
    detail::binder2<Handler, asio::error_code, std::size_t>
      handler(o->handler_, o->ec_, o->result_);
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
  ConstBufferSequence buffers_;
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_SOCKET_SEND_OP_HPP
