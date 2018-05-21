//
// posix/descriptor_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_DESCRIPTOR_BASE_HPP
#define ASIO_POSIX_DESCRIPTOR_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
  || defined(GENERATING_DOCUMENTATION)

#include "asio/detail/io_control.hpp"
#include "asio/detail/socket_option.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace posix {

/// The descriptor_base class is used as a base for the descriptor class as a
/// place to define the associated IO control commands.
class descriptor_base
{
public:
  /// Wait types.
  /**
   * For use with descriptor::wait() and descriptor::async_wait().
   */
  enum wait_type
  {
    /// Wait for a descriptor to become ready to read.
    wait_read,

    /// Wait for a descriptor to become ready to write.
    wait_write,

    /// Wait for a descriptor to have error conditions pending.
    wait_error
  };

  /// IO control command to get the amount of data that can be read without
  /// blocking.
  /**
   * Implements the FIONREAD IO control command.
   *
   * @par Example
   * @code
   * asio::posix::stream_descriptor descriptor(io_context); 
   * ...
   * asio::descriptor_base::bytes_readable command(true);
   * descriptor.io_control(command);
   * std::size_t bytes_readable = command.get();
   * @endcode
   *
   * @par Concepts:
   * IoControlCommand.
   */
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined bytes_readable;
#else
  typedef asio::detail::io_control::bytes_readable bytes_readable;
#endif

protected:
  /// Protected destructor to prevent deletion through this type.
  ~descriptor_base()
  {
  }
};

} // namespace posix
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_POSIX_DESCRIPTOR_BASE_HPP
