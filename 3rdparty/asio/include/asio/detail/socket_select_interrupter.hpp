//
// detail/socket_select_interrupter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_SOCKET_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class socket_select_interrupter
{
public:
  // Constructor.
  ASIO_DECL socket_select_interrupter();

  // Destructor.
  ASIO_DECL ~socket_select_interrupter();

  // Recreate the interrupter's descriptors. Used after a fork.
  ASIO_DECL void recreate();

  // Interrupt the select call.
  ASIO_DECL void interrupt();

  // Reset the select interrupt. Returns true if the call was interrupted.
  ASIO_DECL bool reset();

  // Get the read descriptor to be passed to select.
  socket_type read_descriptor() const
  {
    return read_descriptor_;
  }

private:
  // Open the descriptors. Throws on error.
  ASIO_DECL void open_descriptors();

  // Close the descriptors.
  ASIO_DECL void close_descriptors();

  // The read end of a connection used to interrupt the select call. This file
  // descriptor is passed to select such that when it is time to stop, a single
  // byte will be written on the other end of the connection and this
  // descriptor will become readable.
  socket_type read_descriptor_;

  // The write end of a connection used to interrupt the select call. A single
  // byte may be written to this to wake up the select which is waiting for the
  // other end to become readable.
  socket_type write_descriptor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/socket_select_interrupter.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)

#endif // !defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_SOCKET_SELECT_INTERRUPTER_HPP
