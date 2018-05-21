//
// detail/impl/eventfd_select_interrupter.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Roelof Naude (roelof.naude at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_EVENTFD_SELECT_INTERRUPTER_IPP
#define ASIO_DETAIL_IMPL_EVENTFD_SELECT_INTERRUPTER_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_EVENTFD)

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
# include <asm/unistd.h>
#else // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
# include <sys/eventfd.h>
#endif // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
#include "asio/detail/cstdint.hpp"
#include "asio/detail/eventfd_select_interrupter.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

eventfd_select_interrupter::eventfd_select_interrupter()
{
  open_descriptors();
}

void eventfd_select_interrupter::open_descriptors()
{
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
  write_descriptor_ = read_descriptor_ = syscall(__NR_eventfd, 0);
  if (read_descriptor_ != -1)
  {
    ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
    ::fcntl(read_descriptor_, F_SETFD, FD_CLOEXEC);
  }
#else // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
# if defined(EFD_CLOEXEC) && defined(EFD_NONBLOCK)
  write_descriptor_ = read_descriptor_ =
    ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
# else // defined(EFD_CLOEXEC) && defined(EFD_NONBLOCK)
  errno = EINVAL;
  write_descriptor_ = read_descriptor_ = -1;
# endif // defined(EFD_CLOEXEC) && defined(EFD_NONBLOCK)
  if (read_descriptor_ == -1 && errno == EINVAL)
  {
    write_descriptor_ = read_descriptor_ = ::eventfd(0, 0);
    if (read_descriptor_ != -1)
    {
      ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
      ::fcntl(read_descriptor_, F_SETFD, FD_CLOEXEC);
    }
  }
#endif // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8

  if (read_descriptor_ == -1)
  {
    int pipe_fds[2];
    if (pipe(pipe_fds) == 0)
    {
      read_descriptor_ = pipe_fds[0];
      ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
      ::fcntl(read_descriptor_, F_SETFD, FD_CLOEXEC);
      write_descriptor_ = pipe_fds[1];
      ::fcntl(write_descriptor_, F_SETFL, O_NONBLOCK);
      ::fcntl(write_descriptor_, F_SETFD, FD_CLOEXEC);
    }
    else
    {
      asio::error_code ec(errno,
          asio::error::get_system_category());
      asio::detail::throw_error(ec, "eventfd_select_interrupter");
    }
  }
}

eventfd_select_interrupter::~eventfd_select_interrupter()
{
  close_descriptors();
}

void eventfd_select_interrupter::close_descriptors()
{
  if (write_descriptor_ != -1 && write_descriptor_ != read_descriptor_)
    ::close(write_descriptor_);
  if (read_descriptor_ != -1)
    ::close(read_descriptor_);
}

void eventfd_select_interrupter::recreate()
{
  close_descriptors();

  write_descriptor_ = -1;
  read_descriptor_ = -1;

  open_descriptors();
}

void eventfd_select_interrupter::interrupt()
{
  uint64_t counter(1UL);
  int result = ::write(write_descriptor_, &counter, sizeof(uint64_t));
  (void)result;
}

bool eventfd_select_interrupter::reset()
{
  if (write_descriptor_ == read_descriptor_)
  {
    for (;;)
    {
      // Only perform one read. The kernel maintains an atomic counter.
      uint64_t counter(0);
      errno = 0;
      int bytes_read = ::read(read_descriptor_, &counter, sizeof(uint64_t));
      if (bytes_read < 0 && errno == EINTR)
        continue;
      bool was_interrupted = (bytes_read > 0);
      return was_interrupted;
    }
  }
  else
  {
    for (;;)
    {
      // Clear all data from the pipe.
      char data[1024];
      int bytes_read = ::read(read_descriptor_, data, sizeof(data));
      if (bytes_read < 0 && errno == EINTR)
        continue;
      bool was_interrupted = (bytes_read > 0);
      while (bytes_read == sizeof(data))
        bytes_read = ::read(read_descriptor_, data, sizeof(data));
      return was_interrupted;
    }
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_EVENTFD)

#endif // ASIO_DETAIL_IMPL_EVENTFD_SELECT_INTERRUPTER_IPP
