//
// detail/winrt_ssocket_service_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/winrt_async_manager.hpp"
#include "asio/detail/winrt_socket_recv_op.hpp"
#include "asio/detail/winrt_socket_send_op.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class winrt_ssocket_service_base
{
public:
  // The native type of a socket.
  typedef Windows::Networking::Sockets::StreamSocket^ native_handle_type;

  // The implementation type of the socket.
  struct base_implementation_type
  {
    // Default constructor.
    base_implementation_type()
      : socket_(nullptr),
        next_(0),
        prev_(0)
    {
    }

    // The underlying native socket.
    native_handle_type socket_;

    // Pointers to adjacent socket implementations in linked list.
    base_implementation_type* next_;
    base_implementation_type* prev_;
  };

  // Constructor.
  ASIO_DECL winrt_ssocket_service_base(
      asio::io_context& io_context);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void base_shutdown();

  // Construct a new socket implementation.
  ASIO_DECL void construct(base_implementation_type&);

  // Move-construct a new socket implementation.
  ASIO_DECL void base_move_construct(base_implementation_type& impl,
      base_implementation_type& other_impl);

  // Move-assign from another socket implementation.
  ASIO_DECL void base_move_assign(base_implementation_type& impl,
      winrt_ssocket_service_base& other_service,
      base_implementation_type& other_impl);

  // Destroy a socket implementation.
  ASIO_DECL void destroy(base_implementation_type& impl);

  // Determine whether the socket is open.
  bool is_open(const base_implementation_type& impl) const
  {
    return impl.socket_ != nullptr;
  }

  // Destroy a socket implementation.
  ASIO_DECL asio::error_code close(
      base_implementation_type& impl, asio::error_code& ec);

  // Release ownership of the socket.
  ASIO_DECL native_handle_type release(
      base_implementation_type& impl, asio::error_code& ec);

  // Get the native socket representation.
  native_handle_type native_handle(base_implementation_type& impl)
  {
    return impl.socket_;
  }

  // Cancel all operations associated with the socket.
  asio::error_code cancel(base_implementation_type&,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const base_implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return false;
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const base_implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(base_implementation_type&,
      IO_Control_Command&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the socket.
  bool non_blocking(const base_implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the socket.
  asio::error_code non_blocking(base_implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the native socket implementation.
  bool native_non_blocking(const base_implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the native socket implementation.
  asio::error_code native_non_blocking(base_implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Disable sends or receives on the socket.
  asio::error_code shutdown(base_implementation_type&,
      socket_base::shutdown_type, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Send the given data to the peer.
  template <typename ConstBufferSequence>
  std::size_t send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return do_send(impl,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::first(buffers), flags, ec);
  }

  // Wait until data can be sent without blocking.
  std::size_t send(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef winrt_socket_send_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((io_context_.context(),
          *p.p, "socket", &impl, 0, "async_send"));

    start_send_op(impl,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::first(buffers),
        flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.get_io_context().post(
        detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Receive some data from the peer. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  std::size_t receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return do_receive(impl,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::first(buffers), flags, ec);
  }

  // Wait until data can be received without blocking.
  std::size_t receive(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef winrt_socket_recv_op<MutableBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((io_context_.context(),
          *p.p, "socket", &impl, 0, "async_receive"));

    start_receive_op(impl,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::first(buffers),
        flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.get_io_context().post(
        detail::bind_handler(handler, ec, bytes_transferred));
  }

protected:
  // Helper function to obtain endpoints associated with the connection.
  ASIO_DECL std::size_t do_get_endpoint(
      const base_implementation_type& impl, bool local,
      void* addr, std::size_t addr_len, asio::error_code& ec) const;

  // Helper function to set a socket option.
  ASIO_DECL asio::error_code do_set_option(
      base_implementation_type& impl,
      int level, int optname, const void* optval,
      std::size_t optlen, asio::error_code& ec);

  // Helper function to get a socket option.
  ASIO_DECL void do_get_option(
      const base_implementation_type& impl,
      int level, int optname, void* optval,
      std::size_t* optlen, asio::error_code& ec) const;

  // Helper function to perform a synchronous connect.
  ASIO_DECL asio::error_code do_connect(
      base_implementation_type& impl,
      const void* addr, asio::error_code& ec);

  // Helper function to start an asynchronous connect.
  ASIO_DECL void start_connect_op(
      base_implementation_type& impl, const void* addr,
      winrt_async_op<void>* op, bool is_continuation);

  // Helper function to perform a synchronous send.
  ASIO_DECL std::size_t do_send(
      base_implementation_type& impl, const asio::const_buffer& data,
      socket_base::message_flags flags, asio::error_code& ec);

  // Helper function to start an asynchronous send.
  ASIO_DECL void start_send_op(base_implementation_type& impl,
      const asio::const_buffer& data, socket_base::message_flags flags,
      winrt_async_op<unsigned int>* op, bool is_continuation);

  // Helper function to perform a synchronous receive.
  ASIO_DECL std::size_t do_receive(
      base_implementation_type& impl, const asio::mutable_buffer& data,
      socket_base::message_flags flags, asio::error_code& ec);

  // Helper function to start an asynchronous receive.
  ASIO_DECL void start_receive_op(base_implementation_type& impl,
      const asio::mutable_buffer& data, socket_base::message_flags flags,
      winrt_async_op<Windows::Storage::Streams::IBuffer^>* op,
      bool is_continuation);

  // The io_context implementation used for delivering completions.
  io_context_impl& io_context_;

  // The manager that keeps track of outstanding operations.
  winrt_async_manager& async_manager_;

  // Mutex to protect access to the linked list of implementations. 
  asio::detail::mutex mutex_;

  // The head of a linked list of all implementations.
  base_implementation_type* impl_list_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/winrt_ssocket_service_base.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP
