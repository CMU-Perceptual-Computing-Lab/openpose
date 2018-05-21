//
// detail/executor_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_EXECUTOR_OP_HPP
#define ASIO_DETAIL_EXECUTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/scheduler_operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler, typename Alloc,
    typename Operation = scheduler_operation>
class executor_op : public Operation
{
public:
  ASIO_DEFINE_HANDLER_ALLOCATOR_PTR(executor_op);

  template <typename H>
  executor_op(ASIO_MOVE_ARG(H) h, const Alloc& allocator)
    : Operation(&executor_op::do_complete),
      handler_(ASIO_MOVE_CAST(H)(h)),
      allocator_(allocator)
  {
  }

  static void do_complete(void* owner, Operation* base,
      const asio::error_code& /*ec*/,
      std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    executor_op* o(static_cast<executor_op*>(base));
    Alloc allocator(o->allocator_);
    ptr p = { detail::addressof(allocator), o, o };

    ASIO_HANDLER_COMPLETION((*o));

    // Make a copy of the handler so that the memory can be deallocated before
    // the upcall is made. Even if we're not about to make an upcall, a
    // sub-object of the handler may be the true owner of the memory associated
    // with the handler. Consequently, a local copy of the handler is required
    // to ensure that any owning sub-object remains valid until after we have
    // deallocated the memory here.
    Handler handler(ASIO_MOVE_CAST(Handler)(o->handler_));
    p.reset();

    // Make the upcall if required.
    if (owner)
    {
      fenced_block b(fenced_block::half);
      ASIO_HANDLER_INVOCATION_BEGIN(());
      asio_handler_invoke_helpers::invoke(handler, handler);
      ASIO_HANDLER_INVOCATION_END;
    }
  }

private:
  Handler handler_;
  Alloc allocator_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_EXECUTOR_OP_HPP
