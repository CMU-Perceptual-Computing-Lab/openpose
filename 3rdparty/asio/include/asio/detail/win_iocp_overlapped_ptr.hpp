//
// detail/win_iocp_overlapped_ptr.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP
#define ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/io_context.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/win_iocp_overlapped_op.hpp"
#include "asio/detail/win_iocp_io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Wraps a handler to create an OVERLAPPED object for use with overlapped I/O.
class win_iocp_overlapped_ptr
  : private noncopyable
{
public:
  // Construct an empty win_iocp_overlapped_ptr.
  win_iocp_overlapped_ptr()
    : ptr_(0),
      iocp_service_(0)
  {
  }

  // Construct an win_iocp_overlapped_ptr to contain the specified handler.
  template <typename Handler>
  explicit win_iocp_overlapped_ptr(
      asio::io_context& io_context, ASIO_MOVE_ARG(Handler) handler)
    : ptr_(0),
      iocp_service_(0)
  {
    this->reset(io_context, ASIO_MOVE_CAST(Handler)(handler));
  }

  // Destructor automatically frees the OVERLAPPED object unless released.
  ~win_iocp_overlapped_ptr()
  {
    reset();
  }

  // Reset to empty.
  void reset()
  {
    if (ptr_)
    {
      ptr_->destroy();
      ptr_ = 0;
      iocp_service_->work_finished();
      iocp_service_ = 0;
    }
  }

  // Reset to contain the specified handler, freeing any current OVERLAPPED
  // object.
  template <typename Handler>
  void reset(asio::io_context& io_context, Handler handler)
  {
    typedef win_iocp_overlapped_op<Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(handler);

    ASIO_HANDLER_CREATION((io_context, *p.p,
          "io_context", &io_context.impl_, 0, "overlapped"));

    io_context.impl_.work_started();
    reset();
    ptr_ = p.p;
    p.v = p.p = 0;
    iocp_service_ = &io_context.impl_;
  }

  // Get the contained OVERLAPPED object.
  OVERLAPPED* get()
  {
    return ptr_;
  }

  // Get the contained OVERLAPPED object.
  const OVERLAPPED* get() const
  {
    return ptr_;
  }

  // Release ownership of the OVERLAPPED object.
  OVERLAPPED* release()
  {
    if (ptr_)
      iocp_service_->on_pending(ptr_);

    OVERLAPPED* tmp = ptr_;
    ptr_ = 0;
    iocp_service_ = 0;
    return tmp;
  }

  // Post completion notification for overlapped operation. Releases ownership.
  void complete(const asio::error_code& ec,
      std::size_t bytes_transferred)
  {
    if (ptr_)
    {
      iocp_service_->on_completion(ptr_, ec,
          static_cast<DWORD>(bytes_transferred));
      ptr_ = 0;
      iocp_service_ = 0;
    }
  }

private:
  win_iocp_operation* ptr_;
  win_iocp_io_context* iocp_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP
