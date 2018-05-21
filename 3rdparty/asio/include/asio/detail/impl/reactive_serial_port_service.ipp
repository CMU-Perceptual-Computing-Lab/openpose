//
// detail/impl/reactive_serial_port_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Rep Invariant Systems, Inc. (info@repinvariant.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_REACTIVE_SERIAL_PORT_SERVICE_IPP
#define ASIO_DETAIL_IMPL_REACTIVE_SERIAL_PORT_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_SERIAL_PORT)
#if !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)

#include <cstring>
#include "asio/detail/reactive_serial_port_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

reactive_serial_port_service::reactive_serial_port_service(
    asio::io_context& io_context)
  : service_base<reactive_serial_port_service>(io_context),
    descriptor_service_(io_context)
{
}

void reactive_serial_port_service::shutdown()
{
  descriptor_service_.shutdown();
}

asio::error_code reactive_serial_port_service::open(
    reactive_serial_port_service::implementation_type& impl,
    const std::string& device, asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  descriptor_ops::state_type state = 0;
  int fd = descriptor_ops::open(device.c_str(),
      O_RDWR | O_NONBLOCK | O_NOCTTY, ec);
  if (fd < 0)
    return ec;

  int s = descriptor_ops::fcntl(fd, F_GETFL, ec);
  if (s >= 0)
    s = descriptor_ops::fcntl(fd, F_SETFL, s | O_NONBLOCK, ec);
  if (s < 0)
  {
    asio::error_code ignored_ec;
    descriptor_ops::close(fd, state, ignored_ec);
    return ec;
  }

  // Set up default serial port options.
  termios ios;
  errno = 0;
  s = descriptor_ops::error_wrapper(::tcgetattr(fd, &ios), ec);
  if (s >= 0)
  {
#if defined(_BSD_SOURCE) || defined(_DEFAULT_SOURCE)
    ::cfmakeraw(&ios);
#else
    ios.c_iflag &= ~(IGNBRK | BRKINT | PARMRK
        | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    ios.c_oflag &= ~OPOST;
    ios.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    ios.c_cflag &= ~(CSIZE | PARENB);
    ios.c_cflag |= CS8;
#endif
    ios.c_iflag |= IGNPAR;
    ios.c_cflag |= CREAD | CLOCAL;
    errno = 0;
    s = descriptor_ops::error_wrapper(::tcsetattr(fd, TCSANOW, &ios), ec);
  }
  if (s < 0)
  {
    asio::error_code ignored_ec;
    descriptor_ops::close(fd, state, ignored_ec);
    return ec;
  }

  // We're done. Take ownership of the serial port descriptor.
  if (descriptor_service_.assign(impl, fd, ec))
  {
    asio::error_code ignored_ec;
    descriptor_ops::close(fd, state, ignored_ec);
  }

  return ec;
}

asio::error_code reactive_serial_port_service::do_set_option(
    reactive_serial_port_service::implementation_type& impl,
    reactive_serial_port_service::store_function_type store,
    const void* option, asio::error_code& ec)
{
  termios ios;
  errno = 0;
  descriptor_ops::error_wrapper(::tcgetattr(
        descriptor_service_.native_handle(impl), &ios), ec);
  if (ec)
    return ec;

  if (store(option, ios, ec))
    return ec;

  errno = 0;
  descriptor_ops::error_wrapper(::tcsetattr(
        descriptor_service_.native_handle(impl), TCSANOW, &ios), ec);
  return ec;
}

asio::error_code reactive_serial_port_service::do_get_option(
    const reactive_serial_port_service::implementation_type& impl,
    reactive_serial_port_service::load_function_type load,
    void* option, asio::error_code& ec) const
{
  termios ios;
  errno = 0;
  descriptor_ops::error_wrapper(::tcgetattr(
        descriptor_service_.native_handle(impl), &ios), ec);
  if (ec)
    return ec;

  return load(option, ios, ec);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)
#endif // defined(ASIO_HAS_SERIAL_PORT)

#endif // ASIO_DETAIL_IMPL_REACTIVE_SERIAL_PORT_SERVICE_IPP
