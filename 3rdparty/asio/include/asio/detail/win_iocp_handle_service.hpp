//
// detail/win_iocp_handle_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Rep Invariant Systems, Inc. (info@repinvariant.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/cstdint.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/win_iocp_handle_read_op.hpp"
#include "asio/detail/win_iocp_handle_write_op.hpp"
#include "asio/detail/win_iocp_io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_iocp_handle_service :
  public service_base<win_iocp_handle_service>
{
public:
  // The native type of a stream handle.
  typedef HANDLE native_handle_type;

  // The implementation type of the stream handle.
  class implementation_type
  {
  public:
    // Default constructor.
    implementation_type()
      : handle_(INVALID_HANDLE_VALUE),
        safe_cancellation_thread_id_(0),
        next_(0),
        prev_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class win_iocp_handle_service;

    // The native stream handle representation.
    native_handle_type handle_;

    // The ID of the thread from which it is safe to cancel asynchronous
    // operations. 0 means no asynchronous operations have been started yet.
    // ~0 means asynchronous operations have been started from more than one
    // thread, and cancellation is not supported for the handle.
    DWORD safe_cancellation_thread_id_;

    // Pointers to adjacent handle implementations in linked list.
    implementation_type* next_;
    implementation_type* prev_;
  };

  ASIO_DECL win_iocp_handle_service(asio::io_context& io_context);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Construct a new handle implementation.
  ASIO_DECL void construct(implementation_type& impl);

  // Move-construct a new handle implementation.
  ASIO_DECL void move_construct(implementation_type& impl,
      implementation_type& other_impl);

  // Move-assign from another handle implementation.
  ASIO_DECL void move_assign(implementation_type& impl,
      win_iocp_handle_service& other_service,
      implementation_type& other_impl);

  // Destroy a handle implementation.
  ASIO_DECL void destroy(implementation_type& impl);

  // Assign a native handle to a handle implementation.
  ASIO_DECL asio::error_code assign(implementation_type& impl,
      const native_handle_type& handle, asio::error_code& ec);

  // Determine whether the handle is open.
  bool is_open(const implementation_type& impl) const
  {
    return impl.handle_ != INVALID_HANDLE_VALUE;
  }

  // Destroy a handle implementation.
  ASIO_DECL asio::error_code close(implementation_type& impl,
      asio::error_code& ec);

  // Get the native handle representation.
  native_handle_type native_handle(const implementation_type& impl) const
  {
    return impl.handle_;
  }

  // Cancel all operations associated with the handle.
  ASIO_DECL asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec);

  // Write the given data. Returns the number of bytes written.
  template <typename ConstBufferSequence>
  size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    return write_some_at(impl, 0, buffers, ec);
  }

  // Write the given data at the specified offset. Returns the number of bytes
  // written.
  template <typename ConstBufferSequence>
  size_t write_some_at(implementation_type& impl, uint64_t offset,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    asio::const_buffer buffer =
      buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence>::first(buffers);

    return do_write(impl, offset, buffer, ec);
  }

  // Start an asynchronous write. The data being written must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, Handler& handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_handle_write_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
          reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some"));

    start_write_op(impl, 0,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::first(buffers), p.p);
    p.v = p.p = 0;
  }

  // Start an asynchronous write at a specified offset. The data being written
  // must be valid for the lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_write_some_at(implementation_type& impl, uint64_t offset,
      const ConstBufferSequence& buffers, Handler& handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_handle_write_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
          reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some_at"));

    start_write_op(impl, offset,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::first(buffers), p.p);
    p.v = p.p = 0;
  }

  // Read some data. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    return read_some_at(impl, 0, buffers, ec);
  }

  // Read some data at a specified offset. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t read_some_at(implementation_type& impl, uint64_t offset,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    asio::mutable_buffer buffer =
      buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence>::first(buffers);

    return do_read(impl, offset, buffer, ec);
  }

  // Start an asynchronous read. The buffer for the data being received must be
  // valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, Handler& handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_handle_read_op<MutableBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
          reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some"));

    start_read_op(impl, 0,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::first(buffers), p.p);
    p.v = p.p = 0;
  }

  // Start an asynchronous read at a specified offset. The buffer for the data
  // being received must be valid for the lifetime of the asynchronous
  // operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_read_some_at(implementation_type& impl, uint64_t offset,
      const MutableBufferSequence& buffers, Handler& handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_handle_read_op<MutableBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
          reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some_at"));

    start_read_op(impl, offset,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::first(buffers), p.p);
    p.v = p.p = 0;
  }

private:
  // Prevent the use of the null_buffers type with this service.
  size_t write_some(implementation_type& impl,
      const null_buffers& buffers, asio::error_code& ec);
  size_t write_some_at(implementation_type& impl, uint64_t offset,
      const null_buffers& buffers, asio::error_code& ec);
  template <typename Handler>
  void async_write_some(implementation_type& impl,
      const null_buffers& buffers, Handler& handler);
  template <typename Handler>
  void async_write_some_at(implementation_type& impl, uint64_t offset,
      const null_buffers& buffers, Handler& handler);
  size_t read_some(implementation_type& impl,
      const null_buffers& buffers, asio::error_code& ec);
  size_t read_some_at(implementation_type& impl, uint64_t offset,
      const null_buffers& buffers, asio::error_code& ec);
  template <typename Handler>
  void async_read_some(implementation_type& impl,
      const null_buffers& buffers, Handler& handler);
  template <typename Handler>
  void async_read_some_at(implementation_type& impl, uint64_t offset,
      const null_buffers& buffers, Handler& handler);

  // Helper class for waiting for synchronous operations to complete.
  class overlapped_wrapper;

  // Helper function to perform a synchronous write operation.
  ASIO_DECL size_t do_write(implementation_type& impl,
      uint64_t offset, const asio::const_buffer& buffer,
      asio::error_code& ec);

  // Helper function to start a write operation.
  ASIO_DECL void start_write_op(implementation_type& impl,
      uint64_t offset, const asio::const_buffer& buffer,
      operation* op);

  // Helper function to perform a synchronous write operation.
  ASIO_DECL size_t do_read(implementation_type& impl,
      uint64_t offset, const asio::mutable_buffer& buffer,
      asio::error_code& ec);

  // Helper function to start a read operation.
  ASIO_DECL void start_read_op(implementation_type& impl,
      uint64_t offset, const asio::mutable_buffer& buffer,
      operation* op);

  // Update the ID of the thread from which cancellation is safe.
  ASIO_DECL void update_cancellation_thread_id(implementation_type& impl);

  // Helper function to close a handle when the associated object is being
  // destroyed.
  ASIO_DECL void close_for_destruction(implementation_type& impl);

  // The IOCP service used for running asynchronous operations and dispatching
  // handlers.
  win_iocp_io_context& iocp_service_;

  // Mutex to protect access to the linked list of implementations.
  mutex mutex_;

  // The head of a linked list of all implementations.
  implementation_type* impl_list_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_iocp_handle_service.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
