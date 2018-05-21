//
// detail/buffer_resize_guard.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_BUFFER_RESIZE_GUARD_HPP
#define ASIO_DETAIL_BUFFER_RESIZE_GUARD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/limits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Helper class to manage buffer resizing in an exception safe way.
template <typename Buffer>
class buffer_resize_guard
{
public:
  // Constructor.
  buffer_resize_guard(Buffer& buffer)
    : buffer_(buffer),
      old_size_(buffer.size())
  {
  }

  // Destructor rolls back the buffer resize unless commit was called.
  ~buffer_resize_guard()
  {
    if (old_size_ != (std::numeric_limits<size_t>::max)())
    {
      buffer_.resize(old_size_);
    }
  }

  // Commit the resize transaction.
  void commit()
  {
    old_size_ = (std::numeric_limits<size_t>::max)();
  }

private:
  // The buffer being managed.
  Buffer& buffer_;

  // The size of the buffer at the time the guard was constructed.
  size_t old_size_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BUFFER_RESIZE_GUARD_HPP
