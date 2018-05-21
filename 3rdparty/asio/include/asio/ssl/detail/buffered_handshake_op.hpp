//
// ssl/detail/buffered_handshake_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_BUFFERED_HANDSHAKE_OP_HPP
#define ASIO_SSL_DETAIL_BUFFERED_HANDSHAKE_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/ssl/detail/engine.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename ConstBufferSequence>
class buffered_handshake_op
{
public:
  buffered_handshake_op(stream_base::handshake_type type,
      const ConstBufferSequence& buffers)
    : type_(type),
      buffers_(buffers),
      total_buffer_size_(asio::buffer_size(buffers_))
  {
  }

  engine::want operator()(engine& eng,
      asio::error_code& ec,
      std::size_t& bytes_transferred) const
  {
    return this->process(eng, ec, bytes_transferred,
        asio::buffer_sequence_begin(buffers_),
        asio::buffer_sequence_end(buffers_));
  }

  template <typename Handler>
  void call_handler(Handler& handler,
      const asio::error_code& ec,
      const std::size_t& bytes_transferred) const
  {
    handler(ec, bytes_transferred);
  }

private:
  template <typename Iterator>
  engine::want process(engine& eng,
      asio::error_code& ec,
      std::size_t& bytes_transferred,
      Iterator begin, Iterator end) const
  {
    Iterator iter = begin;
    std::size_t accumulated_size = 0;

    for (;;)
    {
      engine::want want = eng.handshake(type_, ec);
      if (want != engine::want_input_and_retry
          || bytes_transferred == total_buffer_size_)
        return want;

      // Find the next buffer piece to be fed to the engine.
      while (iter != end)
      {
        const_buffer buffer(*iter);

        // Skip over any buffers which have already been consumed by the engine.
        if (bytes_transferred >= accumulated_size + buffer.size())
        {
          accumulated_size += buffer.size();
          ++iter;
          continue;
        }

        // The current buffer may have been partially consumed by the engine on
        // a previous iteration. If so, adjust the buffer to point to the
        // unused portion.
        if (bytes_transferred > accumulated_size)
          buffer = buffer + (bytes_transferred - accumulated_size);

        // Pass the buffer to the engine, and update the bytes transferred to
        // reflect the total number of bytes consumed so far.
        bytes_transferred += buffer.size();
        buffer = eng.put_input(buffer);
        bytes_transferred -= buffer.size();
        break;
      }
    }
  }

  stream_base::handshake_type type_;
  ConstBufferSequence buffers_;
  std::size_t total_buffer_size_;
};

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_BUFFERED_HANDSHAKE_OP_HPP
