//
// detail/socket_holder.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_HOLDER_HPP
#define ASIO_DETAIL_SOCKET_HOLDER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Implement the resource acquisition is initialisation idiom for sockets.
class socket_holder
  : private noncopyable
{
public:
  // Construct as an uninitialised socket.
  socket_holder()
    : socket_(invalid_socket)
  {
  }

  // Construct to take ownership of the specified socket.
  explicit socket_holder(socket_type s)
    : socket_(s)
  {
  }

  // Destructor.
  ~socket_holder()
  {
    if (socket_ != invalid_socket)
    {
      asio::error_code ec;
      socket_ops::state_type state = 0;
      socket_ops::close(socket_, state, true, ec);
    }
  }

  // Get the underlying socket.
  socket_type get() const
  {
    return socket_;
  }

  // Reset to an uninitialised socket.
  void reset()
  {
    if (socket_ != invalid_socket)
    {
      asio::error_code ec;
      socket_ops::state_type state = 0;
      socket_ops::close(socket_, state, true, ec);
      socket_ = invalid_socket;
    }
  }

  // Reset to take ownership of the specified socket.
  void reset(socket_type s)
  {
    reset();
    socket_ = s;
  }

  // Release ownership of the socket.
  socket_type release()
  {
    socket_type tmp = socket_;
    socket_ = invalid_socket;
    return tmp;
  }

private:
  // The underlying socket.
  socket_type socket_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_HOLDER_HPP
