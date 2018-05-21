//
// impl/read_until.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_READ_UNTIL_HPP
#define ASIO_IMPL_READ_UNTIL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include "asio/associated_allocator.hpp"
#include "asio/associated_executor.hpp"
#include "asio/buffer.hpp"
#include "asio/buffers_iterator.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/throw_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename SyncReadStream, typename DynamicBuffer>
inline std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers, char delim)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read_until(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers), delim, ec);
  asio::detail::throw_error(ec, "read_until");
  return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    char delim, asio::error_code& ec)
{
  typename decay<DynamicBuffer>::type b(
      ASIO_MOVE_CAST(DynamicBuffer)(buffers));

  std::size_t search_position = 0;
  for (;;)
  {
    // Determine the range of the data to be searched.
    typedef typename DynamicBuffer::const_buffers_type buffers_type;
    typedef buffers_iterator<buffers_type> iterator;
    buffers_type data_buffers = b.data();
    iterator begin = iterator::begin(data_buffers);
    iterator start_pos = begin + search_position;
    iterator end = iterator::end(data_buffers);

    // Look for a match.
    iterator iter = std::find(start_pos, end, delim);
    if (iter != end)
    {
      // Found a match. We're done.
      ec = asio::error_code();
      return iter - begin + 1;
    }
    else
    {
      // No match. Next search can start with the new data.
      search_position = end - begin;
    }

    // Check if buffer is full.
    if (b.size() == b.max_size())
    {
      ec = error::not_found;
      return 0;
    }

    // Need more data.
    std::size_t bytes_to_read = std::min<std::size_t>(
          std::max<std::size_t>(512, b.capacity() - b.size()),
          std::min<std::size_t>(65536, b.max_size() - b.size()));
    b.commit(s.read_some(b.prepare(bytes_to_read), ec));
    if (ec)
      return 0;
  }
}

template <typename SyncReadStream, typename DynamicBuffer>
inline std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read_until(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers), delim, ec);
  asio::detail::throw_error(ec, "read_until");
  return bytes_transferred;
}

namespace detail
{
  // Algorithm that finds a subsequence of equal values in a sequence. Returns
  // (iterator,true) if a full match was found, in which case the iterator
  // points to the beginning of the match. Returns (iterator,false) if a
  // partial match was found at the end of the first sequence, in which case
  // the iterator points to the beginning of the partial match. Returns
  // (last1,false) if no full or partial match was found.
  template <typename Iterator1, typename Iterator2>
  std::pair<Iterator1, bool> partial_search(
      Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
  {
    for (Iterator1 iter1 = first1; iter1 != last1; ++iter1)
    {
      Iterator1 test_iter1 = iter1;
      Iterator2 test_iter2 = first2;
      for (;; ++test_iter1, ++test_iter2)
      {
        if (test_iter2 == last2)
          return std::make_pair(iter1, true);
        if (test_iter1 == last1)
        {
          if (test_iter2 != first2)
            return std::make_pair(iter1, false);
          else
            break;
        }
        if (*test_iter1 != *test_iter2)
          break;
      }
    }
    return std::make_pair(last1, false);
  }
} // namespace detail

template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim, asio::error_code& ec)
{
  typename decay<DynamicBuffer>::type b(
      ASIO_MOVE_CAST(DynamicBuffer)(buffers));

  std::size_t search_position = 0;
  for (;;)
  {
    // Determine the range of the data to be searched.
    typedef typename DynamicBuffer::const_buffers_type buffers_type;
    typedef buffers_iterator<buffers_type> iterator;
    buffers_type data_buffers = b.data();
    iterator begin = iterator::begin(data_buffers);
    iterator start_pos = begin + search_position;
    iterator end = iterator::end(data_buffers);

    // Look for a match.
    std::pair<iterator, bool> result = detail::partial_search(
        start_pos, end, delim.begin(), delim.end());
    if (result.first != end)
    {
      if (result.second)
      {
        // Full match. We're done.
        ec = asio::error_code();
        return result.first - begin + delim.length();
      }
      else
      {
        // Partial match. Next search needs to start from beginning of match.
        search_position = result.first - begin;
      }
    }
    else
    {
      // No match. Next search can start with the new data.
      search_position = end - begin;
    }

    // Check if buffer is full.
    if (b.size() == b.max_size())
    {
      ec = error::not_found;
      return 0;
    }

    // Need more data.
    std::size_t bytes_to_read = std::min<std::size_t>(
          std::max<std::size_t>(512, b.capacity() - b.size()),
          std::min<std::size_t>(65536, b.max_size() - b.size()));
    b.commit(s.read_some(b.prepare(bytes_to_read), ec));
    if (ec)
      return 0;
  }
}

#if !defined(ASIO_NO_EXTENSIONS)
#if defined(ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename DynamicBuffer>
inline std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read_until(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers), expr, ec);
  asio::detail::throw_error(ec, "read_until");
  return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr, asio::error_code& ec)
{
  typename decay<DynamicBuffer>::type b(
      ASIO_MOVE_CAST(DynamicBuffer)(buffers));

  std::size_t search_position = 0;
  for (;;)
  {
    // Determine the range of the data to be searched.
    typedef typename DynamicBuffer::const_buffers_type buffers_type;
    typedef buffers_iterator<buffers_type> iterator;
    buffers_type data_buffers = b.data();
    iterator begin = iterator::begin(data_buffers);
    iterator start_pos = begin + search_position;
    iterator end = iterator::end(data_buffers);

    // Look for a match.
    boost::match_results<iterator,
      typename std::vector<boost::sub_match<iterator> >::allocator_type>
        match_results;
    if (regex_search(start_pos, end, match_results, expr,
          boost::match_default | boost::match_partial))
    {
      if (match_results[0].matched)
      {
        // Full match. We're done.
        ec = asio::error_code();
        return match_results[0].second - begin;
      }
      else
      {
        // Partial match. Next search needs to start from beginning of match.
        search_position = match_results[0].first - begin;
      }
    }
    else
    {
      // No match. Next search can start with the new data.
      search_position = end - begin;
    }

    // Check if buffer is full.
    if (b.size() == b.max_size())
    {
      ec = error::not_found;
      return 0;
    }

    // Need more data.
    std::size_t bytes_to_read = read_size_helper(b, 65536);
    b.commit(s.read_some(b.prepare(bytes_to_read), ec));
    if (ec)
      return 0;
  }
}

#endif // defined(ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream,
    typename DynamicBuffer, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  asio::error_code ec;
  std::size_t bytes_transferred = read_until(s,
      ASIO_MOVE_CAST(DynamicBuffer)(buffers),
      match_condition, ec);
  asio::detail::throw_error(ec, "read_until");
  return bytes_transferred;
}

template <typename SyncReadStream,
    typename DynamicBuffer, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition, asio::error_code& ec,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  typename decay<DynamicBuffer>::type b(
      ASIO_MOVE_CAST(DynamicBuffer)(buffers));

  std::size_t search_position = 0;
  for (;;)
  {
    // Determine the range of the data to be searched.
    typedef typename DynamicBuffer::const_buffers_type buffers_type;
    typedef buffers_iterator<buffers_type> iterator;
    buffers_type data_buffers = b.data();
    iterator begin = iterator::begin(data_buffers);
    iterator start_pos = begin + search_position;
    iterator end = iterator::end(data_buffers);

    // Look for a match.
    std::pair<iterator, bool> result = match_condition(start_pos, end);
    if (result.second)
    {
      // Full match. We're done.
      ec = asio::error_code();
      return result.first - begin;
    }
    else if (result.first != end)
    {
      // Partial match. Next search needs to start from beginning of match.
      search_position = result.first - begin;
    }
    else
    {
      // No match. Next search can start with the new data.
      search_position = end - begin;
    }

    // Check if buffer is full.
    if (b.size() == b.max_size())
    {
      ec = error::not_found;
      return 0;
    }

    // Need more data.
    std::size_t bytes_to_read = std::min<std::size_t>(
          std::max<std::size_t>(512, b.capacity() - b.size()),
          std::min<std::size_t>(65536, b.max_size() - b.size()));
    b.commit(s.read_some(b.prepare(bytes_to_read), ec));
    if (ec)
      return 0;
  }
}

#if !defined(ASIO_NO_IOSTREAM)

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), delim);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim,
    asio::error_code& ec)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), delim, ec);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), delim);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim, asio::error_code& ec)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), delim, ec);
}

#if defined(ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), expr);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    asio::error_code& ec)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), expr, ec);
}

#endif // defined(ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename Allocator, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, MatchCondition match_condition,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), match_condition);
}

template <typename SyncReadStream, typename Allocator, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    MatchCondition match_condition, asio::error_code& ec,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  return read_until(s, basic_streambuf_ref<Allocator>(b), match_condition, ec);
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

namespace detail
{
  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  class read_until_delim_op
  {
  public:
    template <typename BufferSequence>
    read_until_delim_op(AsyncReadStream& stream,
        ASIO_MOVE_ARG(BufferSequence) buffers,
        char delim, ReadHandler& handler)
      : stream_(stream),
        buffers_(ASIO_MOVE_CAST(BufferSequence)(buffers)),
        delim_(delim),
        start_(0),
        search_position_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_until_delim_op(const read_until_delim_op& other)
      : stream_(other.stream_),
        buffers_(other.buffers_),
        delim_(other.delim_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(other.handler_)
    {
    }

    read_until_delim_op(read_until_delim_op&& other)
      : stream_(other.stream_),
        buffers_(ASIO_MOVE_CAST(DynamicBuffer)(other.buffers_)),
        delim_(other.delim_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
      std::size_t bytes_to_read;
      switch (start_ = start)
      {
      case 1:
        for (;;)
        {
          {
            // Determine the range of the data to be searched.
            typedef typename DynamicBuffer::const_buffers_type
              buffers_type;
            typedef buffers_iterator<buffers_type> iterator;
            buffers_type data_buffers = buffers_.data();
            iterator begin = iterator::begin(data_buffers);
            iterator start_pos = begin + search_position_;
            iterator end = iterator::end(data_buffers);

            // Look for a match.
            iterator iter = std::find(start_pos, end, delim_);
            if (iter != end)
            {
              // Found a match. We're done.
              search_position_ = iter - begin + 1;
              bytes_to_read = 0;
            }

            // No match yet. Check if buffer is full.
            else if (buffers_.size() == buffers_.max_size())
            {
              search_position_ = not_found;
              bytes_to_read = 0;
            }

            // Need to read some more data.
            else
            {
              // Next search can start with the new data.
              search_position_ = end - begin;
              bytes_to_read = std::min<std::size_t>(
                    std::max<std::size_t>(512,
                      buffers_.capacity() - buffers_.size()),
                    std::min<std::size_t>(65536,
                      buffers_.max_size() - buffers_.size()));
            }
          }

          // Check if we're done.
          if (!start && bytes_to_read == 0)
            break;

          // Start a new asynchronous read operation to obtain more data.
          stream_.async_read_some(buffers_.prepare(bytes_to_read),
              ASIO_MOVE_CAST(read_until_delim_op)(*this));
          return; default:
          buffers_.commit(bytes_transferred);
          if (ec || bytes_transferred == 0)
            break;
        }

        const asio::error_code result_ec =
          (search_position_ == not_found)
          ? error::not_found : ec;

        const std::size_t result_n =
          (ec || search_position_ == not_found)
          ? 0 : search_position_;

        handler_(result_ec, result_n);
      }
    }

  //private:
    AsyncReadStream& stream_;
    DynamicBuffer buffers_;
    char delim_;
    int start_;
    std::size_t search_position_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_until_delim_op<AsyncReadStream,
      DynamicBuffer, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename DynamicBuffer,
    typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_until_delim_op<AsyncReadStream,
      DynamicBuffer, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_until_delim_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    char delim, ASIO_MOVE_ARG(ReadHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::read_until_delim_op<AsyncReadStream,
    typename decay<DynamicBuffer>::type,
      ASIO_HANDLER_TYPE(ReadHandler,
        void (asio::error_code, std::size_t))>(
          s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
            delim, init.completion_handler)(asio::error_code(), 0, 1);

  return init.result.get();
}

namespace detail
{
  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  class read_until_delim_string_op
  {
  public:
    template <typename BufferSequence>
    read_until_delim_string_op(AsyncReadStream& stream,
        ASIO_MOVE_ARG(BufferSequence) buffers,
        const std::string& delim, ReadHandler& handler)
      : stream_(stream),
        buffers_(ASIO_MOVE_CAST(BufferSequence)(buffers)),
        delim_(delim),
        start_(0),
        search_position_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_until_delim_string_op(const read_until_delim_string_op& other)
      : stream_(other.stream_),
        buffers_(other.buffers_),
        delim_(other.delim_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(other.handler_)
    {
    }

    read_until_delim_string_op(read_until_delim_string_op&& other)
      : stream_(other.stream_),
        buffers_(ASIO_MOVE_CAST(DynamicBuffer)(other.buffers_)),
        delim_(ASIO_MOVE_CAST(std::string)(other.delim_)),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
      std::size_t bytes_to_read;
      switch (start_ = start)
      {
      case 1:
        for (;;)
        {
          {
            // Determine the range of the data to be searched.
            typedef typename DynamicBuffer::const_buffers_type
              buffers_type;
            typedef buffers_iterator<buffers_type> iterator;
            buffers_type data_buffers = buffers_.data();
            iterator begin = iterator::begin(data_buffers);
            iterator start_pos = begin + search_position_;
            iterator end = iterator::end(data_buffers);

            // Look for a match.
            std::pair<iterator, bool> result = detail::partial_search(
                start_pos, end, delim_.begin(), delim_.end());
            if (result.first != end && result.second)
            {
              // Full match. We're done.
              search_position_ = result.first - begin + delim_.length();
              bytes_to_read = 0;
            }

            // No match yet. Check if buffer is full.
            else if (buffers_.size() == buffers_.max_size())
            {
              search_position_ = not_found;
              bytes_to_read = 0;
            }

            // Need to read some more data.
            else
            {
              if (result.first != end)
              {
                // Partial match. Next search needs to start from beginning of
                // match.
                search_position_ = result.first - begin;
              }
              else
              {
                // Next search can start with the new data.
                search_position_ = end - begin;
              }

              bytes_to_read = std::min<std::size_t>(
                    std::max<std::size_t>(512,
                      buffers_.capacity() - buffers_.size()),
                    std::min<std::size_t>(65536,
                      buffers_.max_size() - buffers_.size()));
            }
          }

          // Check if we're done.
          if (!start && bytes_to_read == 0)
            break;

          // Start a new asynchronous read operation to obtain more data.
          stream_.async_read_some(buffers_.prepare(bytes_to_read),
              ASIO_MOVE_CAST(read_until_delim_string_op)(*this));
          return; default:
          buffers_.commit(bytes_transferred);
          if (ec || bytes_transferred == 0)
            break;
        }

        const asio::error_code result_ec =
          (search_position_ == not_found)
          ? error::not_found : ec;

        const std::size_t result_n =
          (ec || search_position_ == not_found)
          ? 0 : search_position_;

        handler_(result_ec, result_n);
      }
    }

  //private:
    AsyncReadStream& stream_;
    DynamicBuffer buffers_;
    std::string delim_;
    int start_;
    std::size_t search_position_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_until_delim_string_op<AsyncReadStream,
      DynamicBuffer, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename DynamicBuffer,
    typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_until_delim_string_op<AsyncReadStream,
      DynamicBuffer, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_until_delim_string_op<AsyncReadStream,
        DynamicBuffer, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::read_until_delim_string_op<AsyncReadStream,
    typename decay<DynamicBuffer>::type,
      ASIO_HANDLER_TYPE(ReadHandler,
        void (asio::error_code, std::size_t))>(
          s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
            static_cast<std::string>(delim),
              init.completion_handler)(asio::error_code(), 0, 1);

  return init.result.get();
}

#if !defined(ASIO_NO_EXTENSIONS)
#if defined(ASIO_HAS_BOOST_REGEX)

namespace detail
{
  template <typename AsyncReadStream, typename DynamicBuffer,
      typename RegEx, typename ReadHandler>
  class read_until_expr_op
  {
  public:
    template <typename BufferSequence>
    read_until_expr_op(AsyncReadStream& stream,
        ASIO_MOVE_ARG(BufferSequence) buffers,
        const boost::regex& expr, ReadHandler& handler)
      : stream_(stream),
        buffers_(ASIO_MOVE_CAST(BufferSequence)(buffers)),
        expr_(expr),
        start_(0),
        search_position_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_until_expr_op(const read_until_expr_op& other)
      : stream_(other.stream_),
        buffers_(other.buffers_),
        expr_(other.expr_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(other.handler_)
    {
    }

    read_until_expr_op(read_until_expr_op&& other)
      : stream_(other.stream_),
        buffers_(ASIO_MOVE_CAST(DynamicBuffer)(other.buffers_)),
        expr_(other.expr_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
      std::size_t bytes_to_read;
      switch (start_ = start)
      {
      case 1:
        for (;;)
        {
          {
            // Determine the range of the data to be searched.
            typedef typename DynamicBuffer::const_buffers_type
              buffers_type;
            typedef buffers_iterator<buffers_type> iterator;
            buffers_type data_buffers = buffers_.data();
            iterator begin = iterator::begin(data_buffers);
            iterator start_pos = begin + search_position_;
            iterator end = iterator::end(data_buffers);

            // Look for a match.
            boost::match_results<iterator,
              typename std::vector<boost::sub_match<iterator> >::allocator_type>
                match_results;
            bool match = regex_search(start_pos, end, match_results, expr_,
                boost::match_default | boost::match_partial);
            if (match && match_results[0].matched)
            {
              // Full match. We're done.
              search_position_ = match_results[0].second - begin;
              bytes_to_read = 0;
            }

            // No match yet. Check if buffer is full.
            else if (buffers_.size() == buffers_.max_size())
            {
              search_position_ = not_found;
              bytes_to_read = 0;
            }

            // Need to read some more data.
            else
            {
              if (match)
              {
                // Partial match. Next search needs to start from beginning of
                // match.
                search_position_ = match_results[0].first - begin;
              }
              else
              {
                // Next search can start with the new data.
                search_position_ = end - begin;
              }

              bytes_to_read = std::min<std::size_t>(
                    std::max<std::size_t>(512,
                      buffers_.capacity() - buffers_.size()),
                    std::min<std::size_t>(65536,
                      buffers_.max_size() - buffers_.size()));
            }
          }

          // Check if we're done.
          if (!start && bytes_to_read == 0)
            break;

          // Start a new asynchronous read operation to obtain more data.
          stream_.async_read_some(buffers_.prepare(bytes_to_read),
              ASIO_MOVE_CAST(read_until_expr_op)(*this));
          return; default:
          buffers_.commit(bytes_transferred);
          if (ec || bytes_transferred == 0)
            break;
        }

        const asio::error_code result_ec =
          (search_position_ == not_found)
          ? error::not_found : ec;

        const std::size_t result_n =
          (ec || search_position_ == not_found)
          ? 0 : search_position_;

        handler_(result_ec, result_n);
      }
    }

  //private:
    AsyncReadStream& stream_;
    DynamicBuffer buffers_;
    RegEx expr_;
    int start_;
    std::size_t search_position_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename RegEx, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename RegEx, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename RegEx, typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename RegEx, typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename RegEx, typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename RegEx, typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_until_expr_op<AsyncReadStream,
      DynamicBuffer, RegEx, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename DynamicBuffer,
    typename RegEx, typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_until_expr_op<AsyncReadStream,
      DynamicBuffer, RegEx, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_until_expr_op<AsyncReadStream,
        DynamicBuffer, RegEx, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::read_until_expr_op<AsyncReadStream,
    typename decay<DynamicBuffer>::type,
      boost::regex, ASIO_HANDLER_TYPE(ReadHandler,
        void (asio::error_code, std::size_t))>(
          s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
            expr, init.completion_handler)(asio::error_code(), 0, 1);

  return init.result.get();
}

#endif // defined(ASIO_HAS_BOOST_REGEX)

namespace detail
{
  template <typename AsyncReadStream, typename DynamicBuffer,
      typename MatchCondition, typename ReadHandler>
  class read_until_match_op
  {
  public:
    template <typename BufferSequence>
    read_until_match_op(AsyncReadStream& stream,
        ASIO_MOVE_ARG(BufferSequence) buffers,
        MatchCondition match_condition, ReadHandler& handler)
      : stream_(stream),
        buffers_(ASIO_MOVE_CAST(BufferSequence)(buffers)),
        match_condition_(match_condition),
        start_(0),
        search_position_(0),
        handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
    {
    }

#if defined(ASIO_HAS_MOVE)
    read_until_match_op(const read_until_match_op& other)
      : stream_(other.stream_),
        buffers_(other.buffers_),
        match_condition_(other.match_condition_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(other.handler_)
    {
    }

    read_until_match_op(read_until_match_op&& other)
      : stream_(other.stream_),
        buffers_(ASIO_MOVE_CAST(DynamicBuffer)(other.buffers_)),
        match_condition_(other.match_condition_),
        start_(other.start_),
        search_position_(other.search_position_),
        handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    void operator()(const asio::error_code& ec,
        std::size_t bytes_transferred, int start = 0)
    {
      const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
      std::size_t bytes_to_read;
      switch (start_ = start)
      {
      case 1:
        for (;;)
        {
          {
            // Determine the range of the data to be searched.
            typedef typename DynamicBuffer::const_buffers_type
              buffers_type;
            typedef buffers_iterator<buffers_type> iterator;
            buffers_type data_buffers = buffers_.data();
            iterator begin = iterator::begin(data_buffers);
            iterator start_pos = begin + search_position_;
            iterator end = iterator::end(data_buffers);

            // Look for a match.
            std::pair<iterator, bool> result = match_condition_(start_pos, end);
            if (result.second)
            {
              // Full match. We're done.
              search_position_ = result.first - begin;
              bytes_to_read = 0;
            }

            // No match yet. Check if buffer is full.
            else if (buffers_.size() == buffers_.max_size())
            {
              search_position_ = not_found;
              bytes_to_read = 0;
            }

            // Need to read some more data.
            else
            {
              if (result.first != end)
              {
                // Partial match. Next search needs to start from beginning of
                // match.
                search_position_ = result.first - begin;
              }
              else
              {
                // Next search can start with the new data.
                search_position_ = end - begin;
              }

              bytes_to_read = std::min<std::size_t>(
                    std::max<std::size_t>(512,
                      buffers_.capacity() - buffers_.size()),
                    std::min<std::size_t>(65536,
                      buffers_.max_size() - buffers_.size()));
            }
          }

          // Check if we're done.
          if (!start && bytes_to_read == 0)
            break;

          // Start a new asynchronous read operation to obtain more data.
          stream_.async_read_some(buffers_.prepare(bytes_to_read),
              ASIO_MOVE_CAST(read_until_match_op)(*this));
          return; default:
          buffers_.commit(bytes_transferred);
          if (ec || bytes_transferred == 0)
            break;
        }

        const asio::error_code result_ec =
          (search_position_ == not_found)
          ? error::not_found : ec;

        const std::size_t result_n =
          (ec || search_position_ == not_found)
          ? 0 : search_position_;

        handler_(result_ec, result_n);
      }
    }

  //private:
    AsyncReadStream& stream_;
    DynamicBuffer buffers_;
    MatchCondition match_condition_;
    int start_;
    std::size_t search_position_;
    ReadHandler handler_;
  };

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename MatchCondition, typename ReadHandler>
  inline void* asio_handler_allocate(std::size_t size,
      read_until_match_op<AsyncReadStream, DynamicBuffer,
        MatchCondition, ReadHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename MatchCondition, typename ReadHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      read_until_match_op<AsyncReadStream, DynamicBuffer,
        MatchCondition, ReadHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename AsyncReadStream, typename DynamicBuffer,
      typename MatchCondition, typename ReadHandler>
  inline bool asio_handler_is_continuation(
      read_until_match_op<AsyncReadStream, DynamicBuffer,
        MatchCondition, ReadHandler>* this_handler)
  {
    return this_handler->start_ == 0 ? true
      : asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename MatchCondition,
      typename ReadHandler>
  inline void asio_handler_invoke(Function& function,
      read_until_match_op<AsyncReadStream, DynamicBuffer,
        MatchCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename AsyncReadStream,
      typename DynamicBuffer, typename MatchCondition,
      typename ReadHandler>
  inline void asio_handler_invoke(const Function& function,
      read_until_match_op<AsyncReadStream, DynamicBuffer,
      MatchCondition, ReadHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename MatchCondition, typename ReadHandler, typename Allocator>
struct associated_allocator<
    detail::read_until_match_op<AsyncReadStream,
      DynamicBuffer, MatchCondition, ReadHandler>,
    Allocator>
{
  typedef typename associated_allocator<ReadHandler, Allocator>::type type;

  static type get(
      const detail::read_until_match_op<AsyncReadStream,
        DynamicBuffer, MatchCondition, ReadHandler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
  }
};

template <typename AsyncReadStream, typename DynamicBuffer,
    typename MatchCondition, typename ReadHandler, typename Executor>
struct associated_executor<
    detail::read_until_match_op<AsyncReadStream,
      DynamicBuffer, MatchCondition, ReadHandler>,
    Executor>
{
  typedef typename associated_executor<ReadHandler, Executor>::type type;

  static type get(
      const detail::read_until_match_op<AsyncReadStream,
        DynamicBuffer, MatchCondition, ReadHandler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer,
    typename MatchCondition, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition, ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a ReadHandler.
  ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

  async_completion<ReadHandler,
    void (asio::error_code, std::size_t)> init(handler);

  detail::read_until_match_op<AsyncReadStream,
    typename decay<DynamicBuffer>::type,
      MatchCondition, ASIO_HANDLER_TYPE(ReadHandler,
        void (asio::error_code, std::size_t))>(
          s, ASIO_MOVE_CAST(DynamicBuffer)(buffers),
            match_condition, init.completion_handler)(
              asio::error_code(), 0, 1);

  return init.result.get();
}

#if !defined(ASIO_NO_IOSTREAM)

template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    char delim, ASIO_MOVE_ARG(ReadHandler) handler)
{
  return async_read_until(s, basic_streambuf_ref<Allocator>(b),
      delim, ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  return async_read_until(s, basic_streambuf_ref<Allocator>(b),
      delim, ASIO_MOVE_CAST(ReadHandler)(handler));
}

#if defined(ASIO_HAS_BOOST_REGEX)

template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    ASIO_MOVE_ARG(ReadHandler) handler)
{
  return async_read_until(s, basic_streambuf_ref<Allocator>(b),
      expr, ASIO_MOVE_CAST(ReadHandler)(handler));
}

#endif // defined(ASIO_HAS_BOOST_REGEX)

template <typename AsyncReadStream, typename Allocator,
    typename MatchCondition, typename ReadHandler>
inline ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    MatchCondition match_condition, ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
  return async_read_until(s, basic_streambuf_ref<Allocator>(b),
      match_condition, ASIO_MOVE_CAST(ReadHandler)(handler));
}

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_READ_UNTIL_HPP
