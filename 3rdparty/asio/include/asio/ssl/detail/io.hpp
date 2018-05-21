//
// ssl/detail/io.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_IO_HPP
#define ASIO_SSL_DETAIL_IO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/ssl/detail/engine.hpp"
#include "asio/ssl/detail/stream_core.hpp"
#include "asio/write.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename Stream, typename Operation>
std::size_t io(Stream& next_layer, stream_core& core,
    const Operation& op, asio::error_code& ec)
{
  std::size_t bytes_transferred = 0;
  do switch (op(core.engine_, ec, bytes_transferred))
  {
  case engine::want_input_and_retry:

    // If the input buffer is empty then we need to read some more data from
    // the underlying transport.
    if (core.input_.size() == 0)
      core.input_ = asio::buffer(core.input_buffer_,
          next_layer.read_some(core.input_buffer_, ec));

    // Pass the new input data to the engine.
    core.input_ = core.engine_.put_input(core.input_);

    // Try the operation again.
    continue;

  case engine::want_output_and_retry:

    // Get output data from the engine and write it to the underlying
    // transport.
    asio::write(next_layer,
        core.engine_.get_output(core.output_buffer_), ec);

    // Try the operation again.
    continue;

  case engine::want_output:

    // Get output data from the engine and write it to the underlying
    // transport.
    asio::write(next_layer,
        core.engine_.get_output(core.output_buffer_), ec);

    // Operation is complete. Return result to caller.
    core.engine_.map_error_code(ec);
    return bytes_transferred;

  default:

    // Operation is complete. Return result to caller.
    core.engine_.map_error_code(ec);
    return bytes_transferred;

  } while (!ec);

  // Operation failed. Return result to caller.
  core.engine_.map_error_code(ec);
  return 0;
}

template <typename Stream, typename Operation, typename Handler>
class io_op
{
public:
  io_op(Stream& next_layer, stream_core& core,
      const Operation& op, Handler& handler)
    : next_layer_(next_layer),
      core_(core),
      op_(op),
      start_(0),
      want_(engine::want_nothing),
      bytes_transferred_(0),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
  }

#if defined(ASIO_HAS_MOVE)
  io_op(const io_op& other)
    : next_layer_(other.next_layer_),
      core_(other.core_),
      op_(other.op_),
      start_(other.start_),
      want_(other.want_),
      ec_(other.ec_),
      bytes_transferred_(other.bytes_transferred_),
      handler_(other.handler_)
  {
  }

  io_op(io_op&& other)
    : next_layer_(other.next_layer_),
      core_(other.core_),
      op_(ASIO_MOVE_CAST(Operation)(other.op_)),
      start_(other.start_),
      want_(other.want_),
      ec_(other.ec_),
      bytes_transferred_(other.bytes_transferred_),
      handler_(ASIO_MOVE_CAST(Handler)(other.handler_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  void operator()(asio::error_code ec,
      std::size_t bytes_transferred = ~std::size_t(0), int start = 0)
  {
    switch (start_ = start)
    {
    case 1: // Called after at least one async operation.
      do
      {
        switch (want_ = op_(core_.engine_, ec_, bytes_transferred_))
        {
        case engine::want_input_and_retry:

          // If the input buffer already has data in it we can pass it to the
          // engine and then retry the operation immediately.
          if (core_.input_.size() != 0)
          {
            core_.input_ = core_.engine_.put_input(core_.input_);
            continue;
          }

          // The engine wants more data to be read from input. However, we
          // cannot allow more than one read operation at a time on the
          // underlying transport. The pending_read_ timer's expiry is set to
          // pos_infin if a read is in progress, and neg_infin otherwise.
          if (core_.expiry(core_.pending_read_) == core_.neg_infin())
          {
            // Prevent other read operations from being started.
            core_.pending_read_.expires_at(core_.pos_infin());

            // Start reading some data from the underlying transport.
            next_layer_.async_read_some(
                asio::buffer(core_.input_buffer_),
                ASIO_MOVE_CAST(io_op)(*this));
          }
          else
          {
            // Wait until the current read operation completes.
            core_.pending_read_.async_wait(ASIO_MOVE_CAST(io_op)(*this));
          }

          // Yield control until asynchronous operation completes. Control
          // resumes at the "default:" label below.
          return;

        case engine::want_output_and_retry:
        case engine::want_output:

          // The engine wants some data to be written to the output. However, we
          // cannot allow more than one write operation at a time on the
          // underlying transport. The pending_write_ timer's expiry is set to
          // pos_infin if a write is in progress, and neg_infin otherwise.
          if (core_.expiry(core_.pending_write_) == core_.neg_infin())
          {
            // Prevent other write operations from being started.
            core_.pending_write_.expires_at(core_.pos_infin());

            // Start writing all the data to the underlying transport.
            asio::async_write(next_layer_,
                core_.engine_.get_output(core_.output_buffer_),
                ASIO_MOVE_CAST(io_op)(*this));
          }
          else
          {
            // Wait until the current write operation completes.
            core_.pending_write_.async_wait(ASIO_MOVE_CAST(io_op)(*this));
          }

          // Yield control until asynchronous operation completes. Control
          // resumes at the "default:" label below.
          return;

        default:

          // The SSL operation is done and we can invoke the handler, but we
          // have to keep in mind that this function might be being called from
          // the async operation's initiating function. In this case we're not
          // allowed to call the handler directly. Instead, issue a zero-sized
          // read so the handler runs "as-if" posted using io_context::post().
          if (start)
          {
            next_layer_.async_read_some(
                asio::buffer(core_.input_buffer_, 0),
                ASIO_MOVE_CAST(io_op)(*this));

            // Yield control until asynchronous operation completes. Control
            // resumes at the "default:" label below.
            return;
          }
          else
          {
            // Continue on to run handler directly.
            break;
          }
        }

        default:
        if (bytes_transferred == ~std::size_t(0))
          bytes_transferred = 0; // Timer cancellation, no data transferred.
        else if (!ec_)
          ec_ = ec;

        switch (want_)
        {
        case engine::want_input_and_retry:

          // Add received data to the engine's input.
          core_.input_ = asio::buffer(
              core_.input_buffer_, bytes_transferred);
          core_.input_ = core_.engine_.put_input(core_.input_);

          // Release any waiting read operations.
          core_.pending_read_.expires_at(core_.neg_infin());

          // Try the operation again.
          continue;

        case engine::want_output_and_retry:

          // Release any waiting write operations.
          core_.pending_write_.expires_at(core_.neg_infin());

          // Try the operation again.
          continue;

        case engine::want_output:

          // Release any waiting write operations.
          core_.pending_write_.expires_at(core_.neg_infin());

          // Fall through to call handler.

        default:

          // Pass the result to the handler.
          op_.call_handler(handler_,
              core_.engine_.map_error_code(ec_),
              ec_ ? 0 : bytes_transferred_);

          // Our work here is done.
          return;
        }
      } while (!ec_);

      // Operation failed. Pass the result to the handler.
      op_.call_handler(handler_, core_.engine_.map_error_code(ec_), 0);
    }
  }

//private:
  Stream& next_layer_;
  stream_core& core_;
  Operation op_;
  int start_;
  engine::want want_;
  asio::error_code ec_;
  std::size_t bytes_transferred_;
  Handler handler_;
};

template <typename Stream, typename Operation, typename Handler>
inline void* asio_handler_allocate(std::size_t size,
    io_op<Stream, Operation, Handler>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Stream, typename Operation, typename Handler>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    io_op<Stream, Operation, Handler>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Stream, typename Operation, typename Handler>
inline bool asio_handler_is_continuation(
    io_op<Stream, Operation, Handler>* this_handler)
{
  return this_handler->start_ == 0 ? true
    : asio_handler_cont_helpers::is_continuation(this_handler->handler_);
}

template <typename Function, typename Stream,
    typename Operation, typename Handler>
inline void asio_handler_invoke(Function& function,
    io_op<Stream, Operation, Handler>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Stream,
    typename Operation, typename Handler>
inline void asio_handler_invoke(const Function& function,
    io_op<Stream, Operation, Handler>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Stream, typename Operation, typename Handler>
inline void async_io(Stream& next_layer, stream_core& core,
    const Operation& op, Handler& handler)
{
  io_op<Stream, Operation, Handler>(
    next_layer, core, op, handler)(
      asio::error_code(), 0, 1);
}

} // namespace detail
} // namespace ssl

template <typename Stream, typename Operation,
    typename Handler, typename Allocator>
struct associated_allocator<
    ssl::detail::io_op<Stream, Operation, Handler>, Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const ssl::detail::io_op<Stream, Operation, Handler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Stream, typename Operation,
    typename Handler, typename Executor>
struct associated_executor<
    ssl::detail::io_op<Stream, Operation, Handler>, Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const ssl::detail::io_op<Stream, Operation, Handler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_IO_HPP
