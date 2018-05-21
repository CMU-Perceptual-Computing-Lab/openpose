//
// seq_packet_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SEQ_PACKET_SOCKET_SERVICE_HPP
#define ASIO_SEQ_PACKET_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/null_socket_service.hpp"
#elif defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_socket_service.hpp"
#else
# include "asio/detail/reactive_socket_service.hpp"
#endif

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default service implementation for a sequenced packet socket.
template <typename Protocol>
class seq_packet_socket_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<
      seq_packet_socket_service<Protocol> >
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_WINDOWS_RUNTIME)
  typedef detail::null_socket_service<Protocol> service_impl_type;
#elif defined(ASIO_HAS_IOCP)
  typedef detail::win_iocp_socket_service<Protocol> service_impl_type;
#else
  typedef detail::reactive_socket_service<Protocol> service_impl_type;
#endif

public:
  /// The type of a sequenced packet socket implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// The native socket type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef typename service_impl_type::native_handle_type native_handle_type;
#endif

  /// Construct a new sequenced packet socket service for the specified
  /// io_context.
  explicit seq_packet_socket_service(asio::io_context& io_context)
    : asio::detail::service_base<
        seq_packet_socket_service<Protocol> >(io_context),
      service_impl_(io_context)
  {
  }

  /// Construct a new sequenced packet socket implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a new sequenced packet socket implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl)
  {
    service_impl_.move_construct(impl, other_impl);
  }

  /// Move-assign from another sequenced packet socket implementation.
  void move_assign(implementation_type& impl,
      seq_packet_socket_service& other_service,
      implementation_type& other_impl)
  {
    service_impl_.move_assign(impl, other_service.service_impl_, other_impl);
  }

  // All socket services have access to each other's implementations.
  template <typename Protocol1> friend class seq_packet_socket_service;

  /// Move-construct a new sequenced packet socket implementation from another
  /// protocol type.
  template <typename Protocol1>
  void converting_move_construct(implementation_type& impl,
      seq_packet_socket_service<Protocol1>& other_service,
      typename seq_packet_socket_service<
        Protocol1>::implementation_type& other_impl,
      typename enable_if<is_convertible<
        Protocol1, Protocol>::value>::type* = 0)
  {
    service_impl_.template converting_move_construct<Protocol1>(
        impl, other_service.service_impl_, other_impl);
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destroy a sequenced packet socket implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Open a sequenced packet socket.
  ASIO_SYNC_OP_VOID open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (protocol.type() == ASIO_OS_DEF(SOCK_SEQPACKET))
      service_impl_.open(impl, protocol, ec);
    else
      ec = asio::error::invalid_argument;
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Assign an existing native socket to a sequenced packet socket.
  ASIO_SYNC_OP_VOID assign(implementation_type& impl,
      const protocol_type& protocol, const native_handle_type& native_socket,
      asio::error_code& ec)
  {
    service_impl_.assign(impl, protocol, native_socket, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the socket is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Close a sequenced packet socket implementation.
  ASIO_SYNC_OP_VOID close(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.close(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Release ownership of the underlying socket.
  native_handle_type release(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.release(impl, ec);
  }

  /// Get the native socket implementation.
  native_handle_type native_handle(implementation_type& impl)
  {
    return service_impl_.native_handle(impl);
  }

  /// Cancel all asynchronous operations associated with the socket.
  ASIO_SYNC_OP_VOID cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.cancel(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return service_impl_.at_mark(impl, ec);
  }

  /// Determine the number of bytes available for reading.
  std::size_t available(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return service_impl_.available(impl, ec);
  }

  /// Bind the sequenced packet socket to the specified local endpoint.
  ASIO_SYNC_OP_VOID bind(implementation_type& impl,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    service_impl_.bind(impl, endpoint, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Connect the sequenced packet socket to the specified endpoint.
  ASIO_SYNC_OP_VOID connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, asio::error_code& ec)
  {
    service_impl_.connect(impl, peer_endpoint, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Start an asynchronous connect.
  template <typename ConnectHandler>
  ASIO_INITFN_RESULT_TYPE(ConnectHandler,
      void (asio::error_code))
  async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint,
      ASIO_MOVE_ARG(ConnectHandler) handler)
  {
    async_completion<ConnectHandler,
      void (asio::error_code)> init(handler);

    service_impl_.async_connect(impl, peer_endpoint, init.completion_handler);

    return init.result.get();
  }

  /// Set a socket option.
  template <typename SettableSocketOption>
  ASIO_SYNC_OP_VOID set_option(implementation_type& impl,
      const SettableSocketOption& option, asio::error_code& ec)
  {
    service_impl_.set_option(impl, option, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get a socket option.
  template <typename GettableSocketOption>
  ASIO_SYNC_OP_VOID get_option(const implementation_type& impl,
      GettableSocketOption& option, asio::error_code& ec) const
  {
    service_impl_.get_option(impl, option, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Perform an IO control command on the socket.
  template <typename IoControlCommand>
  ASIO_SYNC_OP_VOID io_control(implementation_type& impl,
      IoControlCommand& command, asio::error_code& ec)
  {
    service_impl_.io_control(impl, command, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Gets the non-blocking mode of the socket.
  bool non_blocking(const implementation_type& impl) const
  {
    return service_impl_.non_blocking(impl);
  }

  /// Sets the non-blocking mode of the socket.
  ASIO_SYNC_OP_VOID non_blocking(implementation_type& impl,
      bool mode, asio::error_code& ec)
  {
    service_impl_.non_blocking(impl, mode, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Gets the non-blocking mode of the native socket implementation.
  bool native_non_blocking(const implementation_type& impl) const
  {
    return service_impl_.native_non_blocking(impl);
  }

  /// Sets the non-blocking mode of the native socket implementation.
  ASIO_SYNC_OP_VOID native_non_blocking(implementation_type& impl,
      bool mode, asio::error_code& ec)
  {
    service_impl_.native_non_blocking(impl, mode, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return service_impl_.local_endpoint(impl, ec);
  }

  /// Get the remote endpoint.
  endpoint_type remote_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return service_impl_.remote_endpoint(impl, ec);
  }

  /// Disable sends or receives on the socket.
  ASIO_SYNC_OP_VOID shutdown(implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec)
  {
    service_impl_.shutdown(impl, what, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Wait for the socket to become ready to read, ready to write, or to have
  /// pending error conditions.
  ASIO_SYNC_OP_VOID wait(implementation_type& impl,
      socket_base::wait_type w, asio::error_code& ec)
  {
    service_impl_.wait(impl, w, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Asynchronously wait for the socket to become ready to read, ready to
  /// write, or to have pending error conditions.
  template <typename WaitHandler>
  ASIO_INITFN_RESULT_TYPE(WaitHandler,
      void (asio::error_code))
  async_wait(implementation_type& impl, socket_base::wait_type w,
      ASIO_MOVE_ARG(WaitHandler) handler)
  {
    async_completion<WaitHandler,
      void (asio::error_code)> init(handler);

    service_impl_.async_wait(impl, w, init.completion_handler);

    return init.result.get();
  }

  /// Send the given data to the peer.
  template <typename ConstBufferSequence>
  std::size_t send(implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return service_impl_.send(impl, buffers, flags, ec);
  }

  /// Start an asynchronous send.
  template <typename ConstBufferSequence, typename WriteHandler>
  ASIO_INITFN_RESULT_TYPE(WriteHandler,
      void (asio::error_code, std::size_t))
  async_send(implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags,
      ASIO_MOVE_ARG(WriteHandler) handler)
  {
    async_completion<WriteHandler,
      void (asio::error_code, std::size_t)> init(handler);

    service_impl_.async_send(impl, buffers, flags, init.completion_handler);

    return init.result.get();
  }

  /// Receive some data from the peer.
  template <typename MutableBufferSequence>
  std::size_t receive(implementation_type& impl,
      const MutableBufferSequence& buffers, socket_base::message_flags in_flags,
      socket_base::message_flags& out_flags, asio::error_code& ec)
  {
    return service_impl_.receive_with_flags(impl,
        buffers, in_flags, out_flags, ec);
  }

  /// Start an asynchronous receive.
  template <typename MutableBufferSequence, typename ReadHandler>
  ASIO_INITFN_RESULT_TYPE(ReadHandler,
      void (asio::error_code, std::size_t))
  async_receive(implementation_type& impl,
      const MutableBufferSequence& buffers, socket_base::message_flags in_flags,
      socket_base::message_flags& out_flags,
      ASIO_MOVE_ARG(ReadHandler) handler)
  {
    async_completion<ReadHandler,
      void (asio::error_code, std::size_t)> init(handler);

    service_impl_.async_receive_with_flags(impl,
        buffers, in_flags, out_flags, init.completion_handler);

    return init.result.get();
  }

private:
  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    service_impl_.shutdown();
  }

  // The platform-specific implementation.
  service_impl_type service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_SEQ_PACKET_SOCKET_SERVICE_HPP
