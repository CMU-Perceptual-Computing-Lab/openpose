//
// detail/null_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class null_socket_service :
  public service_base<null_socket_service<Protocol> >
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  // The native type of a socket.
  typedef int native_handle_type;

  // The implementation type of the socket.
  struct implementation_type
  {
  };

  // Constructor.
  null_socket_service(asio::io_context& io_context)
    : service_base<null_socket_service<Protocol> >(io_context),
      io_context_(io_context)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
  }

  // Construct a new socket implementation.
  void construct(implementation_type&)
  {
  }

  // Move-construct a new socket implementation.
  void move_construct(implementation_type&, implementation_type&)
  {
  }

  // Move-assign from another socket implementation.
  void move_assign(implementation_type&,
      null_socket_service&, implementation_type&)
  {
  }

  // Move-construct a new socket implementation from another protocol type.
  template <typename Protocol1>
  void converting_move_construct(implementation_type&,
      null_socket_service<Protocol1>&,
      typename null_socket_service<Protocol1>::implementation_type&)
  {
  }

  // Destroy a socket implementation.
  void destroy(implementation_type&)
  {
  }

  // Open a new socket implementation.
  asio::error_code open(implementation_type&,
      const protocol_type&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code assign(implementation_type&, const protocol_type&,
      const native_handle_type&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Determine whether the socket is open.
  bool is_open(const implementation_type&) const
  {
    return false;
  }

  // Destroy a socket implementation.
  asio::error_code close(implementation_type&,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Release ownership of the socket.
  native_handle_type release(implementation_type&,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Get the native socket representation.
  native_handle_type native_handle(implementation_type&)
  {
    return 0;
  }

  // Cancel all operations associated with the socket.
  asio::error_code cancel(implementation_type&,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return false;
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Place the socket into the state where it will listen for new connections.
  asio::error_code listen(implementation_type&,
      int, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(implementation_type&,
      IO_Control_Command&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the socket.
  bool non_blocking(const implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the socket.
  asio::error_code non_blocking(implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the native socket implementation.
  bool native_non_blocking(const implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the native socket implementation.
  asio::error_code native_non_blocking(implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Disable sends or receives on the socket.
  asio::error_code shutdown(implementation_type&,
      socket_base::shutdown_type, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Bind the socket to the specified local endpoint.
  asio::error_code bind(implementation_type&,
      const endpoint_type&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code set_option(implementation_type&,
      const Option&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code get_option(const implementation_type&,
      Option&, asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return endpoint_type();
  }

  // Get the remote endpoint.
  endpoint_type remote_endpoint(const implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return endpoint_type();
  }

  // Send the given data to the peer.
  template <typename ConstBufferSequence>
  std::size_t send(implementation_type&, const ConstBufferSequence&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be sent without blocking.
  std::size_t send(implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send(implementation_type&, const ConstBufferSequence&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send(implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Receive some data from the peer. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  std::size_t receive(implementation_type&, const MutableBufferSequence&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be received without blocking.
  std::size_t receive(implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive(implementation_type&, const MutableBufferSequence&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive(implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Receive some data with associated flags. Returns the number of bytes
  // received.
  template <typename MutableBufferSequence>
  std::size_t receive_with_flags(implementation_type&,
      const MutableBufferSequence&, socket_base::message_flags,
      socket_base::message_flags&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be received without blocking.
  std::size_t receive_with_flags(implementation_type&,
      const null_buffers&, socket_base::message_flags,
      socket_base::message_flags&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive_with_flags(implementation_type&,
      const MutableBufferSequence&, socket_base::message_flags,
      socket_base::message_flags&, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive_with_flags(implementation_type&,
      const null_buffers&, socket_base::message_flags,
      socket_base::message_flags&, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename ConstBufferSequence>
  std::size_t send_to(implementation_type&, const ConstBufferSequence&,
      const endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be sent without blocking.
  std::size_t send_to(implementation_type&, const null_buffers&,
      const endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send_to(implementation_type&, const ConstBufferSequence&,
      const endpoint_type&, socket_base::message_flags,
      Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send_to(implementation_type&, const null_buffers&,
      const endpoint_type&, socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename MutableBufferSequence>
  std::size_t receive_from(implementation_type&, const MutableBufferSequence&,
      endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be received without blocking.
  std::size_t receive_from(implementation_type&, const null_buffers&,
      endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive_from(implementation_type&,
      const MutableBufferSequence&, endpoint_type&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive_from(implementation_type&,
      const null_buffers&, endpoint_type&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_context_.post(detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Accept a new connection.
  template <typename Socket>
  asio::error_code accept(implementation_type&,
      Socket&, endpoint_type*, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(implementation_type&, Socket&,
      endpoint_type*, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    io_context_.post(detail::bind_handler(handler, ec));
  }

  // Connect the socket to the specified endpoint.
  asio::error_code connect(implementation_type&,
      const endpoint_type&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Start an asynchronous connect.
  template <typename Handler>
  void async_connect(implementation_type&,
      const endpoint_type&, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    io_context_.post(detail::bind_handler(handler, ec));
  }

private:
  asio::io_context& io_context_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP
