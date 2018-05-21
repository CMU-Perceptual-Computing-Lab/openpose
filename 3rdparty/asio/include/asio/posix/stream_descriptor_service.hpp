//
// posix/stream_descriptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP
#define ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
  || defined(GENERATING_DOCUMENTATION)

#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/detail/reactive_descriptor_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace posix {

/// Default service implementation for a stream descriptor.
class stream_descriptor_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<stream_descriptor_service>
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

private:
  // The type of the platform-specific implementation.
  typedef detail::reactive_descriptor_service service_impl_type;

public:
  /// The type of a stream descriptor implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef service_impl_type::implementation_type implementation_type;
#endif

  /// The native descriptor type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef service_impl_type::native_handle_type native_handle_type;
#endif

  /// Construct a new stream descriptor service for the specified io_context.
  explicit stream_descriptor_service(asio::io_context& io_context)
    : asio::detail::service_base<stream_descriptor_service>(io_context),
      service_impl_(io_context)
  {
  }

  /// Construct a new stream descriptor implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a new stream descriptor implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl)
  {
    service_impl_.move_construct(impl, other_impl);
  }

  /// Move-assign from another stream descriptor implementation.
  void move_assign(implementation_type& impl,
      stream_descriptor_service& other_service,
      implementation_type& other_impl)
  {
    service_impl_.move_assign(impl, other_service.service_impl_, other_impl);
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destroy a stream descriptor implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Assign an existing native descriptor to a stream descriptor.
  ASIO_SYNC_OP_VOID assign(implementation_type& impl,
      const native_handle_type& native_descriptor,
      asio::error_code& ec)
  {
    service_impl_.assign(impl, native_descriptor, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the descriptor is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Close a stream descriptor implementation.
  ASIO_SYNC_OP_VOID close(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.close(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get the native descriptor implementation.
  native_handle_type native_handle(implementation_type& impl)
  {
    return service_impl_.native_handle(impl);
  }

  /// Release ownership of the native descriptor implementation.
  native_handle_type release(implementation_type& impl)
  {
    return service_impl_.release(impl);
  }

  /// Cancel all asynchronous operations associated with the descriptor.
  ASIO_SYNC_OP_VOID cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    service_impl_.cancel(impl, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Perform an IO control command on the descriptor.
  template <typename IoControlCommand>
  ASIO_SYNC_OP_VOID io_control(implementation_type& impl,
      IoControlCommand& command, asio::error_code& ec)
  {
    service_impl_.io_control(impl, command, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Gets the non-blocking mode of the descriptor.
  bool non_blocking(const implementation_type& impl) const
  {
    return service_impl_.non_blocking(impl);
  }

  /// Sets the non-blocking mode of the descriptor.
  ASIO_SYNC_OP_VOID non_blocking(implementation_type& impl,
      bool mode, asio::error_code& ec)
  {
    service_impl_.non_blocking(impl, mode, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Gets the non-blocking mode of the native descriptor implementation.
  bool native_non_blocking(const implementation_type& impl) const
  {
    return service_impl_.native_non_blocking(impl);
  }

  /// Sets the non-blocking mode of the native descriptor implementation.
  ASIO_SYNC_OP_VOID native_non_blocking(implementation_type& impl,
      bool mode, asio::error_code& ec)
  {
    service_impl_.native_non_blocking(impl, mode, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Wait for the descriptor to become ready to read, ready to write, or to
  /// have pending error conditions.
  ASIO_SYNC_OP_VOID wait(implementation_type& impl,
      descriptor_base::wait_type w, asio::error_code& ec)
  {
    service_impl_.wait(impl, w, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Asynchronously wait for the descriptor to become ready to read, ready to
  /// write, or to have pending error conditions.
  template <typename WaitHandler>
  ASIO_INITFN_RESULT_TYPE(WaitHandler,
      void (asio::error_code))
  async_wait(implementation_type& impl, descriptor_base::wait_type w,
      ASIO_MOVE_ARG(WaitHandler) handler)
  {
    async_completion<WaitHandler,
      void (asio::error_code)> init(handler);

    service_impl_.async_wait(impl, w, init.completion_handler);

    return init.result.get();
  }

  /// Write the given data to the stream.
  template <typename ConstBufferSequence>
  std::size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.write_some(impl, buffers, ec);
  }

  /// Start an asynchronous write.
  template <typename ConstBufferSequence, typename WriteHandler>
  ASIO_INITFN_RESULT_TYPE(WriteHandler,
      void (asio::error_code, std::size_t))
  async_write_some(implementation_type& impl,
      const ConstBufferSequence& buffers,
      ASIO_MOVE_ARG(WriteHandler) handler)
  {
    asio::async_completion<WriteHandler,
      void (asio::error_code, std::size_t)> init(handler);

    service_impl_.async_write_some(impl, buffers, init.completion_handler);

    return init.result.get();
  }

  /// Read some data from the stream.
  template <typename MutableBufferSequence>
  std::size_t read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.read_some(impl, buffers, ec);
  }

  /// Start an asynchronous read.
  template <typename MutableBufferSequence, typename ReadHandler>
  ASIO_INITFN_RESULT_TYPE(ReadHandler,
      void (asio::error_code, std::size_t))
  async_read_some(implementation_type& impl,
      const MutableBufferSequence& buffers,
      ASIO_MOVE_ARG(ReadHandler) handler)
  {
    asio::async_completion<ReadHandler,
      void (asio::error_code, std::size_t)> init(handler);

    service_impl_.async_read_some(impl, buffers, init.completion_handler);

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

} // namespace posix
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP
