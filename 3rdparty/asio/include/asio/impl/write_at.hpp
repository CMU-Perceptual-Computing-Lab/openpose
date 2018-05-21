//
// impl/write_at.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_WRITE_AT_HPP
#define ASIO_IMPL_WRITE_AT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/associated_allocator.hpp"
#include "asio/associated_executor.hpp"
#include "asio/buffer.hpp"
#include "asio/completion_condition.hpp"
#include "asio/detail/array_fwd.hpp"
#include "asio/detail/base_from_completion_cond.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/consuming_buffers.hpp"
#include "asio/detail/dependent_type.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/throw_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail
{
  template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
      typename ConstBufferIterator, typename CompletionCondition>
  std::size_t write_at_buffer_sequence(SyncRandomAccessWriteDevice& d,
      uint64_t offset, const ConstBufferSequence& buffers,
      const ConstBufferIterator&, CompletionCondition completion_condition,
      asio::error_code& ec)
  {
    ec = asio::error_code();
    asio::detail::consuming_buffers<const_buffer,
        ConstBufferSequence, ConstBufferIterator> tmp(buffers);
    while (!tmp.empty())
    {
      if (std::size_t max_size = detail::adapt_completion_condition_result(
            completion_condition(ec, tmp.total_consumed())))
      {
        tmp.consume(d.write_some_at(offset + tmp.total_consumed(),
              tmp.prepare(max_size), ec));
      }
      else
        break;
    }
    return tmp.total_consumed();;
  }
} // namespace detail

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  return detail::write_at_buffer_sequence(d, offset, buffers,
      asio::buffer_sequence_begin(buffers), completion_condition, ec);
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, buffers, transfer_all(), ec);
  asio::detail::throw_error(ec, "write_at");
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers,
    asio::error_code& ec)
{
  return write_at(d, offset, buffers, transfer_all(), ec);
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, buffers, completion_condition, ec);
  asio::detail::throw_error(ec, "write_at");
  return bytes_transferred;
}

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

template <typename SyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  std::size_t bytes_transferred = write_at(
      d, offset, b.data(), completion_condition, ec);
  b.consume(bytes_transferred);
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(d, offset, b, transfer_all(), ec);
  asio::detail::throw_error(ec, "write_at");
  return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b,
    asio::error_code& ec)
{
  return write_at(d, offset, b, transfer_all(), ec);
}

template <typename SyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition)
{
  asio::error_code ec;
  std::size_t bytes_transferred = write_at(
      d, offset, b, completion_condition, ec);
  asio::detail::throw_error(ec, "write_at");
  return bytes_transferred;
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

namespace detail
{
  template <typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  class write_at_op
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    write_at_op(AsyncRandomAccessWriteDevice& device,
        uint64_t offset, const ConstBufferSequence& buffers,
        CompletionCondition completion_condition, WriteHandler& handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        device_(device),
        offset_(offset),
        buffers_(buffers),
        start_(0),
        handler_(ASIO_MOVE_CAST(WriteHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    write_at_op(const write_at_op& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        device_(other.device_),
        offset_(other.offset_),
        buffers_(other.buffers_),
        start_(other.start_),
        handler_(other.handler_)
    {
    }

    write_at_op(write_at_op&& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        device_(other.device_),
        offset_(other.offset_),
        buffers_(other.buffers_),
        start_(other.start_),
        handler_(ASIO_MOVE_CAST(WriteHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      std::size_t max_size;
      switch (start_ = start)
      {
        case 1:
        max_size = this->check_for_completion(ec, buffers_.total_consumed());
        do
        {
          device_.async_write_some_at(
              offset_ + buffers_.total_consumed(), buffers_.prepare(max_size),
              ASIO_MOVE_CAST(write_at_op)(*this));
          return; default:
          buffers_.consume(bytes_transferred);
          if ((!ec && bytes_transferred == 0) || buffers_.empty())
            break;
          max_size = this->check_for_completion(ec, buffers_.total_consumed());
        } while (max_size > 0);

        handler_(ec, buffers_.total_consumed());
      }
    }

  //private:
    AsyncRandomAccessWriteDevice& device_;
    uint64_t offset_;
    asio::detail::consuming_buffers<const_buffer,
        ConstBufferSequence, ConstBufferIterator> buffers_;
    int start_;
    WriteHandler handler_;
  };

  template <typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline void* asio_handler_allocate(std::size_t size,
      write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline bool asio_handler_is_continuation(
      write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline void asio_handler_invoke(Function& function,
      write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline void asio_handler_invoke(const Function& function,
      write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
        ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename AsyncRandomAccessWriteDevice,
      typename ConstBufferSequence, typename ConstBufferIterator,
      typename CompletionCondition, typename WriteHandler>
  inline void start_write_at_buffer_sequence_op(AsyncRandomAccessWriteDevice& d,
      uint64_t offset, const ConstBufferSequence& buffers,
      const ConstBufferIterator&, CompletionCondition completion_condition,
      WriteHandler& handler)
  {
    detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
      ConstBufferIterator, CompletionCondition, WriteHandler>(
        d, offset, buffers, completion_condition, handler)(
          asio::error_code(), 0, 1);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncRandomAccessWriteDevice,
    typename ConstBufferSequence, typename ConstBufferIterator,
    typename CompletionCondition, typename WriteHandler, typename Allocator>
struct associated_allocator<
    detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
      ConstBufferIterator, CompletionCondition, WriteHandler>,
    Allocator>
{
  typedef typename associated_allocator<WriteHandler, Allocator>::type type;

  static type get(
      const detail::write_at_op<AsyncRandomAccessWriteDevice,
        ConstBufferSequence, ConstBufferIterator,
        CompletionCondition, WriteHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<WriteHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncRandomAccessWriteDevice,
    typename ConstBufferSequence, typename ConstBufferIterator,
    typename CompletionCondition, typename WriteHandler, typename Executor>
struct associated_executor<
    detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
      ConstBufferIterator, CompletionCondition, WriteHandler>,
    Executor>
{
  typedef typename associated_executor<WriteHandler, Executor>::type type;

  static type get(
      const detail::write_at_op<AsyncRandomAccessWriteDevice,
        ConstBufferSequence, ConstBufferIterator,
        CompletionCondition, WriteHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<WriteHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename CompletionCondition, typename WriteHandler>
inline ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(WriteHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a WriteHandler.
  ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

  async_completion<WriteHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::start_write_at_buffer_sequence_op(d, offset, buffers,
      asio::buffer_sequence_begin(buffers), completion_condition,
      init.completion_handler);

  return init.result.get();
}

template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
    typename WriteHandler>
inline ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
    uint64_t offset, const ConstBufferSequence& buffers,
    ASIO_MOVE_ARG(WriteHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a WriteHandler.
  ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

  async_completion<WriteHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::start_write_at_buffer_sequence_op(d, offset, buffers,
      asio::buffer_sequence_begin(buffers), transfer_all(),
      init.completion_handler);

  return init.result.get();
}

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

namespace detail
{
  template <typename Allocator, typename WriteHandler>
  class write_at_streambuf_op
  {
  public:
    write_at_streambuf_op(
        asio::basic_streambuf<Allocator>& streambuf,
        WriteHandler& handler)
      : streambuf_(streambuf),
        handler_(ASIO_MOVE_CAST(WriteHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    write_at_streambuf_op(const write_at_streambuf_op& other)
      : streambuf_(other.streambuf_),
        handler_(other.handler_)
    {
    }

    write_at_streambuf_op(write_at_streambuf_op&& other)
      : streambuf_(other.streambuf_),
        handler_(ASIO_MOVE_CAST(WriteHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        const std::size_t bytes_transferred)
    {
      streambuf_.consume(bytes_transferred);
      handler_(ec, bytes_transferred);
    }

  //private:
    asio::basic_streambuf<Allocator>& streambuf_;
    WriteHandler handler_;
  };

  template <typename Allocator, typename WriteHandler>
  inline void* asio_handler_allocate(std::size_t size,
      write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename Allocator, typename WriteHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Allocator, typename WriteHandler>
  inline bool asio_handler_is_continuation(
      write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
  {
    return asio_handler_cont_helpers::is_continuation(
        this_handler->handler_);
  }

  template <typename Function, typename Allocator, typename WriteHandler>
  inline void asio_handler_invoke(Function& function,
      write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename Allocator, typename WriteHandler>
  inline void asio_handler_invoke(const Function& function,
      write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Allocator, typename WriteHandler>
  inline write_at_streambuf_op<Allocator, WriteHandler>
  make_write_at_streambuf_op(
      asio::basic_streambuf<Allocator>& b, WriteHandler handler)
  {
    return write_at_streambuf_op<Allocator, WriteHandler>(b, handler);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename Allocator, typename WriteHandler, typename Allocator1>
struct associated_allocator<
    detail::write_at_streambuf_op<Allocator, WriteHandler>,
    Allocator1>
{
  typedef typename associated_allocator<WriteHandler, Allocator1>::type type;

  static type get(
      const detail::write_at_streambuf_op<Allocator, WriteHandler>& h,
      const Allocator1& a = Allocator1()) ASIO_NOEXCEPT
  {
    return associated_allocator<WriteHandler, Allocator1>::get(h.handler_, a);
  }
};

template <typename Executor, typename WriteHandler, typename Executor1>
struct associated_executor<
    detail::write_at_streambuf_op<Executor, WriteHandler>,
    Executor1>
{
  typedef typename associated_executor<WriteHandler, Executor1>::type type;

  static type get(
      const detail::write_at_streambuf_op<Executor, WriteHandler>& h,
      const Executor1& ex = Executor1()) ASIO_NOEXCEPT
  {
    return associated_executor<WriteHandler, Executor1>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncRandomAccessWriteDevice, typename Allocator,
    typename CompletionCondition, typename WriteHandler>
inline ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(WriteHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a WriteHandler.
  ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

  async_completion<WriteHandler,
    void (asio::error_code, std::size_t)> init(handler);

  async_write_at(d, offset, b.data(), completion_condition,
    detail::write_at_streambuf_op<Allocator, ASIO_HANDLER_TYPE(
      WriteHandler, void (asio::error_code, std::size_t))>(
        b, init.completion_handler));

  return init.result.get();
}

template <typename AsyncRandomAccessWriteDevice, typename Allocator,
    typename WriteHandler>
inline ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
    uint64_t offset, asio::basic_streambuf<Allocator>& b,
    ASIO_MOVE_ARG(WriteHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a WriteHandler.
  ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

  async_completion<WriteHandler,
    void (asio::error_code, std::size_t)> init(handler);

  async_write_at(d, offset, b.data(), transfer_all(),
    detail::write_at_streambuf_op<Allocator, ASIO_HANDLER_TYPE(
      WriteHandler, void (asio::error_code, std::size_t))>(
        b, init.completion_handler));

  return init.result.get();
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_WRITE_AT_HPP
