//
// impl/read.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_READ_HPP
#define ASIO_IMPL_READ_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <algorithm>
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
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail
{
  template <typename SyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition>
  std::size_t read_buffer_sequence(SyncReadStream& s,
      const MutableBufferSequence& buffers, const MutableBufferIterator&,
      CompletionCondition completion_condition, asio::error_code& ec)
  {
    ec = asio::error_code();
    asio::detail::consuming_buffers<mutable_buffer,
        MutableBufferSequence, MutableBufferIterator> tmp(buffers);
    while (!tmp.empty())
    {
      if (std::size_t max_size = detail::adapt_completion_condition_result(
            completion_condition(ec, tmp.total_consumed())))
        tmp.consume(s.read_some(tmp.prepare(max_size), ec));
      else
        break;
    }
    return tmp.total_consumed();;
  }
} // namespace detail

template <typename SyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition>
std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition, asio::error_code& ec,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  return detail::read_buffer_sequence(s, buffers,
      asio::buffer_sequence_begin(buffers), completion_condition, ec);
}

template <typename SyncReadStream, typename MutableBufferSequence>
inline std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, buffers, transfer_all(), ec);
  asio::detail::throw_error(ec, "read");
  return bytes_transferred;
}

template <typename SyncReadStream, typename MutableBufferSequence>
inline std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    asio::error_code& ec,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  return read(s, buffers, transfer_all(), ec);
}

template <typename SyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s, buffers, completion_condition, ec);
  asio::detail::throw_error(ec, "read");
  return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer,
    typename CompletionCondition>
std::size_t read(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition, asio::error_code& ec,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  typename decay<DynamicBuffer>::type b(
      ASIO_MOVE_CAST(DynamicBuffer)(buffers));

  ec = asio::error_code();
  std::size_t total_transferred = 0;
  std::size_t max_size = detail::adapt_completion_condition_result(
        completion_condition(ec, total_transferred));
  std::size_t bytes_available = std::min<std::size_t>(
        std::max<std::size_t>(512, b.capacity() - b.size()),
        std::min<std::size_t>(max_size, b.max_size() - b.size()));
  while (bytes_available > 0)
  {
    std::size_t bytes_transferred = s.read_some(b.prepare(bytes_available), ec);
    b.commit(bytes_transferred);
    total_transferred += bytes_transferred;
    max_size = detail::adapt_completion_condition_result(
          completion_condition(ec, total_transferred));
    bytes_available = std::min<std::size_t>(
          std::max<std::size_t>(512, b.capacity() - b.size()),
          std::min<std::size_t>(max_size, b.max_size() - b.size()));
  }
  return total_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer>
inline std::size_t read(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers), transfer_all(), ec);
  asio::detail::throw_error(ec, "read");
  return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer>
inline std::size_t read(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    asio::error_code& ec,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  return read(s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
      transfer_all(), ec);
}

template <typename SyncReadStream, typename DynamicBuffer,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers),
      completion_condition, ec);
  asio::detail::throw_error(ec, "read");
  return bytes_transferred;
}

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

template <typename SyncReadStream, typename Allocator,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, asio::error_code& ec)
{
  return read(s, basic_streambuf_ref<Allocator>(b), completion_condition, ec);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b)
{
  return read(s, basic_streambuf_ref<Allocator>(b));
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    asio::error_code& ec)
{
  return read(s, basic_streambuf_ref<Allocator>(b), ec);
}

template <typename SyncReadStream, typename Allocator,
    typename CompletionCondition>
inline std::size_t read(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition)
{
  return read(s, basic_streambuf_ref<Allocator>(b), completion_condition);
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

namespace detail
{
  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition,
      typename ReadHandler>
  class read_op
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    read_op(AsyncReadStream& stream, const MutableBufferSequence& buffers,
        CompletionCondition completion_condition, ReadHandler& handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        stream_(stream),
        buffers_(buffers),
        start_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_op(const read_op& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        stream_(other.stream_),
        buffers_(other.buffers_),
        start_(other.start_),
        handler_(other.handler_)
    {
    }

    read_op(read_op&& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        stream_(other.stream_),
        buffers_(other.buffers_),
        start_(other.start_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
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
          stream_.async_read_some(buffers_.prepare(max_size),
              ASIO_MOVE_CAST(read_op)(*this));
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
    AsyncReadStream& stream_;
    asio::detail::consuming_buffers<mutable_buffer,
        MutableBufferSequence, MutableBufferIterator> buffers_;
    int start_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition,
      typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_op<AsyncReadStream, MutableBufferSequence, MutableBufferIterator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition,
      typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_op<AsyncReadStream, MutableBufferSequence, MutableBufferIterator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition,
      typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_op<AsyncReadStream, MutableBufferSequence, MutableBufferIterator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename MutableBufferSequence, typename MutableBufferIterator,
      typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_op<AsyncReadStream, MutableBufferSequence, MutableBufferIterator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename MutableBufferSequence, typename MutableBufferIterator,
      typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_op<AsyncReadStream, MutableBufferSequence, MutableBufferIterator,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename MutableBufferSequence,
      typename MutableBufferIterator, typename CompletionCondition,
      typename ReadHandler>
  inline void start_read_buffer_sequence_op(AsyncReadStream& stream,
      const MutableBufferSequence& buffers, const MutableBufferIterator&,
      CompletionCondition completion_condition, ReadHandler& handler)
  {
    detail::read_op<AsyncReadStream, MutableBufferSequence,
      MutableBufferIterator, CompletionCondition, ReadHandler>(
        stream, buffers, completion_condition, handler)(
          asio::error_code(), 0, 1);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename MutableBufferIterator, typename CompletionCondition,
    typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_op<AsyncReadStream, MutableBufferSequence,
      MutableBufferIterator, CompletionCondition, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_op<AsyncReadStream, MutableBufferSequence,
        MutableBufferIterator, CompletionCondition, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename MutableBufferIterator, typename CompletionCondition,
    typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_op<AsyncReadStream, MutableBufferSequence,
      MutableBufferIterator, CompletionCondition, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_op<AsyncReadStream, MutableBufferSequence,
        MutableBufferIterator, CompletionCondition, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename CompletionCondition, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s, const MutableBufferSequence& buffers,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::start_read_buffer_sequence_op(s, buffers,
      asio::buffer_sequence_begin(buffers), completion_condition,
      init.completion_handler);

  return init.result.get();
}

template <typename AsyncReadStream, typename MutableBufferSequence,
    typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s, const MutableBufferSequence& buffers,
    ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<
      is_mutable_buffer_sequence<MutableBufferSequence>::value
    >::type*)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::start_read_buffer_sequence_op(s, buffers,
      asio::buffer_sequence_begin(buffers), transfer_all(),
      init.completion_handler);

  return init.result.get();
}

namespace detail
{
  template <typename AsyncReadStream, typename DynamicBuffer,
      typename CompletionCondition, typename ReadHandler>
  class read_dynbuf_op
    : detail::base_from_completion_cond<CompletionCondition>
  {
  public:
    template <typename BufferSequence>
    read_dynbuf_op(AsyncReadStream& stream,
        ASIO_MOVE_ARG(BufferSequence) buffers,
        CompletionCondition completion_condition, ReadHandler& handler)
      : detail::base_from_completion_cond<
          CompletionCondition>(completion_condition),
        stream_(stream),
        buffers_(ASIO_MOVE_CAST(BufferSequence)(buffers)),
        start_(0),
        total_transferred_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_dynbuf_op(const read_dynbuf_op& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        stream_(other.stream_),
        buffers_(other.buffers_),
        start_(other.start_),
        total_transferred_(other.total_transferred_),
        handler_(other.handler_)
    {
    }

    read_dynbuf_op(read_dynbuf_op&& other)
      : detail::base_from_completion_cond<CompletionCondition>(other),
        stream_(other.stream_),
        buffers_(ASIO_MOVE_CAST(DynamicBuffer)(other.buffers_)),
        start_(other.start_),
        total_transferred_(other.total_transferred_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      std::size_t max_size, bytes_available;
      switch (start_ = start)
      {
        case 1:
        max_size = this->check_for_completion(ec, total_transferred_);
        bytes_available = std::min<std::size_t>(
              std::max<std::size_t>(512,
                buffers_.capacity() - buffers_.size()),
              std::min<std::size_t>(max_size,
                buffers_.max_size() - buffers_.size()));
        for (;;)
        {
          stream_.async_read_some(buffers_.prepare(bytes_available),
              ASIO_MOVE_CAST(read_dynbuf_op)(*this));
          return; default:
          total_transferred_ += bytes_transferred;
          buffers_.commit(bytes_transferred);
          max_size = this->check_for_completion(ec, total_transferred_);
          bytes_available = std::min<std::size_t>(
                std::max<std::size_t>(512,
                  buffers_.capacity() - buffers_.size()),
                std::min<std::size_t>(max_size,
                  buffers_.max_size() - buffers_.size()));
          if ((!ec && bytes_transferred == 0) || bytes_available == 0)
            break;
        }

        handler_(ec, static_cast<const std::size_t&>(total_transferred_));
      }
    }

  //private:
    AsyncReadStream& stream_;
    DynamicBuffer buffers_;
    int start_;
    std::size_t total_transferred_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename CompletionCondition, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_dynbuf_op<AsyncReadStream, DynamicBuffer,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename CompletionCondition, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_dynbuf_op<AsyncReadStream, DynamicBuffer,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename CompletionCondition, typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_dynbuf_op<AsyncReadStream, DynamicBuffer,
        CompletionCondition, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename CompletionCondition,
      typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_dynbuf_op<AsyncReadStream, DynamicBuffer,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename CompletionCondition,
      typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_dynbuf_op<AsyncReadStream, DynamicBuffer,
        CompletionCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename CompletionCondition, typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_dynbuf_op<AsyncReadStream,
      DynamicBuffer, CompletionCondition, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_dynbuf_op<AsyncReadStream,
        DynamicBuffer, CompletionCondition, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename DynamicBuffer,
    typename CompletionCondition, typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_dynbuf_op<AsyncReadStream,
      DynamicBuffer, CompletionCondition, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_dynbuf_op<AsyncReadStream,
        DynamicBuffer, CompletionCondition, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  return async_read(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers),
      transfer_all(), ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename AsyncReadStream, typename DynamicBuffer,
    typename CompletionCondition, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type*)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::read_dynbuf_op<AsyncReadStream,
    typename decay<DynamicBuffer>::type,
      CompletionCondition, ASIO_HANDLER_TYPE(
        ReadHandler, void (asio::error_code, std::size_t))>(
          s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
            completion_condition, init.completion_handler)(
              asio::error_code(), 0, 1);

  return init.result.get();
}

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s, basic_streambuf<Allocator>& b,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  return async_read(s, basic_streambuf_ref<Allocator>(b),
      ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename AsyncReadStream, typename Allocator,
    typename CompletionCondition, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read(AsyncReadStream& s, basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  return async_read(s, basic_streambuf_ref<Allocator>(b),
      completion_condition, ASIO_MOVE_CAST(ReadHandler)(handler));
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_READ_HPP
