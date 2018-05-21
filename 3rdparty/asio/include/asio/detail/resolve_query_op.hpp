//
// detail/resolve_query_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_RESOLVE_QUERY_OP_HPP
#define ASIO_DETAIL_RESOLVE_QUERY_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/ip/basic_resolver_query.hpp"
#include "asio/ip/basic_resolver_results.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/resolve_op.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol, typename Handler>
class resolve_query_op : public resolve_op
{
public:
  ASIO_DEFINE_HANDLER_PTR(resolve_query_op);

  typedef asio::ip::basic_resolver_query<Protocol> query_type;
  typedef asio::ip::basic_resolver_results<Protocol> results_type;

  resolve_query_op(socket_ops::weak_cancel_token_type cancel_token,
      const query_type& query, io_context_impl& ioc, Handler& handler)
    : resolve_op(&resolve_query_op::do_complete),
      cancel_token_(cancel_token),
      query_(query),
      io_context_impl_(ioc),
      handler_(ASIO_MOVE_CAST(Handler)(handler)),
      addrinfo_(0)
  {
    handler_work<Handler>::start(handler_);
  }

  ~resolve_query_op()
  {
    if (addrinfo_)
      socket_ops::freeaddrinfo(addrinfo_);
  }

  static void do_complete(void* owner, operation* base,
      const asio::error_code& /*ec*/,
      std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the operation object.
    resolve_query_op* o(static_cast<resolve_query_op*>(base));
    ptr p = { asio::detail::addressof(o->handler_), o, o };

    if (owner && owner != &o->io_context_impl_)
    {
      // The operation is being run on the worker io_context. Time to perform
      // the resolver operation.
    
      // Perform the blocking host resolution operation.
      socket_ops::background_getaddrinfo(o->cancel_token_,
          o->query_.host_name().c_str(), o->query_.service_name().c_str(),
          o->query_.hints(), &o->addrinfo_, o->ec_);

      // Pass operation back to main io_context for completion.
      o->io_context_impl_.post_deferred_completion(o);
      p.v = p.p = 0;
    }
    else
    {
      // The operation has been returned to the main io_context. The completion
      // handler is ready to be delivered.

      // Take ownership of the operation's outstanding work.
      handler_work<Handler> w(o->handler_);

      ASIO_HANDLER_COMPLETION((*o));

      // Make a copy of the handler so that the memory can be deallocated
      // before the upcall is made. Even if we're not about to make an upcall,
      // a sub-object of the handler may be the true owner of the memory
      // associated with the handler. Consequently, a local copy of the handler
      // is required to ensure that any owning sub-object remains valid until
      // after we have deallocated the memory here.
      detail::binder2<Handler, asio::error_code, results_type>
        handler(o->handler_, o->ec_, results_type());
      p.h = asio::detail::addressof(handler.handler_);
      if (o->addrinfo_)
      {
        handler.arg2_ = results_type::create(o->addrinfo_,
            o->query_.host_name(), o->query_.service_name());
      }
      p.reset();

      if (owner)
      {
        fenced_block b(fenced_block::half);
        ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, "..."));
        w.complete(handler, handler.handler_);
        ASIO_HANDLER_INVOCATION_END;
      }
    }
  }

private:
  socket_ops::weak_cancel_token_type cancel_token_;
  query_type query_;
  io_context_impl& io_context_impl_;
  Handler handler_;
  asio::detail::addrinfo_type* addrinfo_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_RESOLVE_QUERY_OP_HPP
