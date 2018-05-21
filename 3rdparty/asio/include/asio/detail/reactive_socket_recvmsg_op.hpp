//
// detail/reactive_socket_recvmsg_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_RECVMSG_OP_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_RECVMSG_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/socket_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename MutableBufferSequence>
class reactive_socket_recvmsg_op_base : public reactor_op
{
public:
  reactive_socket_recvmsg_op_base(socket_type socket,
      const MutableBufferSequence& buffers, socket_base::message_flags in_flags,
      socket_base::message_flags& out_flags, func_type complete_func)
    : reactor_op(&reactive_socket_recvmsg_op_base::do_perform, complete_func),
      socket_(socket),
      buffers_(buffers),
      in_flags_(in_flags),
      out_flags_(out_flags)
  {
  }

  static status do_perform(reactor_op* base)
  {
    reactive_socket_recvmsg_op_base* o(
        static_cast<reactive_socket_recvmsg_op_base*>(base));

    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(o->buffers_);

    status result = socket_ops::non_blocking_recvmsg(o->socket_,
        bufs.buffers(), bufs.count(),
        o->in_flags_, o->out_flags_,
        o->ec_, o->bytes_transferred_) ? done : not_done;

    ASIO_HANDLER_REACTOR_OPERATION((*o, "non_blocking_recvmsg",
          o->ec_, o->bytes_transferred_));

    return result;
  }

private:
  socket_type socket_;
  MutableBufferSequence buffers_;
  socket_base::message_flags in_flags_;
  socket_base::message_flags& out_flags_;
};

template <typename MutableBufferSequence, typename Handler>
class reactive_socket_recvmsg_op :
  public reactive_socket_recvmsg_op_base<MutableBufferSequence>
{
public:
  ASIO_DEFINE_HANDLER_PTR(reactive_socket_recvmsg_op);

  reactive_socket_recvmsg_op(socket_type socket,
      const MutableBufferSequence& buffers, socket_base::message_flags in_flags,
      socket_base::message_flags& out_flags, Handler& handler)
    : reactive_socket_recvmsg_op_base<MutableBufferSequence>(socket, buffers,
        in_flags, out_flags, &reactive_socket_recvmsg_op::do_complete),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
    handler_work<Handler>::start(handler_);
  }

  static void do_complete(void* owner, operation* base,
      const asio::error_code& /*ec*/,
      std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    reactive_socket_recvmsg_op* o(
        static_cast<reactive_socket_recvmsg_op*>(base));
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
      handler(o->handler_, o->ec_, o->bytes_transferred_);
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

#endif // ASIO_DETAIL_REACTIVE_SOCKET_RECVMSG_OP_HPP
