//
// detail/impl/socket_ops.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_OPS_IPP
#define ASIO_DETAIL_SOCKET_OPS_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <new>
#include "asio/detail/assert.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/error.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
# include <codecvt>
# include <locale>
# include <string>
#endif // defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__) \
  || defined(__MACH__) && defined(__APPLE__)
# if defined(ASIO_HAS_PTHREADS)
#  include <pthread.h>
# endif // defined(ASIO_HAS_PTHREADS)
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
       // || defined(__MACH__) && defined(__APPLE__)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace socket_ops {

#if !defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
struct msghdr { int msg_namelen; };
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

#if defined(__hpux)
// HP-UX doesn't declare these functions extern "C", so they are declared again
// here to avoid linker errors about undefined symbols.
extern "C" char* if_indextoname(unsigned int, char*);
extern "C" unsigned int if_nametoindex(const char*);
#endif // defined(__hpux)

#endif // !defined(ASIO_WINDOWS_RUNTIME)

inline void clear_last_error()
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  WSASetLastError(0);
#else
  errno = 0;
#endif
}

#if !defined(ASIO_WINDOWS_RUNTIME)

template <typename ReturnType>
inline ReturnType error_wrapper(ReturnType return_value,
    asio::error_code& ec)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  ec = asio::error_code(WSAGetLastError(),
      asio::error::get_system_category());
#else
  ec = asio::error_code(errno,
      asio::error::get_system_category());
#endif
  return return_value;
}

template <typename SockLenType>
inline socket_type call_accept(SockLenType msghdr::*,
    socket_type s, socket_addr_type* addr, std::size_t* addrlen)
{
  SockLenType tmp_addrlen = addrlen ? (SockLenType)*addrlen : 0;
  socket_type result = ::accept(s, addr, addrlen ? &tmp_addrlen : 0);
  if (addrlen)
    *addrlen = (std::size_t)tmp_addrlen;
  return result;
}

socket_type accept(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return invalid_socket;
  }

  clear_last_error();

  socket_type new_s = error_wrapper(call_accept(
        &msghdr::msg_namelen, s, addr, addrlen), ec);
  if (new_s == invalid_socket)
    return new_s;

#if defined(__MACH__) && defined(__APPLE__) || defined(__FreeBSD__)
  int optval = 1;
  int result = error_wrapper(::setsockopt(new_s,
        SOL_SOCKET, SO_NOSIGPIPE, &optval, sizeof(optval)), ec);
  if (result != 0)
  {
    ::close(new_s);
    return invalid_socket;
  }
#endif

  ec = asio::error_code();
  return new_s;
}

socket_type sync_accept(socket_type s, state_type state,
    socket_addr_type* addr, std::size_t* addrlen, asio::error_code& ec)
{
  // Accept a socket.
  for (;;)
  {
    // Try to complete the operation without blocking.
    socket_type new_socket = socket_ops::accept(s, addr, addrlen, ec);

    // Check if operation succeeded.
    if (new_socket != invalid_socket)
      return new_socket;

    // Operation failed.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
    {
      if (state & user_set_non_blocking)
        return invalid_socket;
      // Fall through to retry operation.
    }
    else if (ec == asio::error::connection_aborted)
    {
      if (state & enable_connection_aborted)
        return invalid_socket;
      // Fall through to retry operation.
    }
#if defined(EPROTO)
    else if (ec.value() == EPROTO)
    {
      if (state & enable_connection_aborted)
        return invalid_socket;
      // Fall through to retry operation.
    }
#endif // defined(EPROTO)
    else
      return invalid_socket;

    // Wait for socket to become ready.
    if (socket_ops::poll_read(s, 0, -1, ec) < 0)
      return invalid_socket;
  }
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_accept(socket_type s,
    void* output_buffer, DWORD address_length,
    socket_addr_type* addr, std::size_t* addrlen,
    socket_type new_socket, asio::error_code& ec)
{
  // Map non-portable errors to their portable counterparts.
  if (ec.value() == ERROR_NETNAME_DELETED)
    ec = asio::error::connection_aborted;

  if (!ec)
  {
    // Get the address of the peer.
    if (addr && addrlen)
    {
      LPSOCKADDR local_addr = 0;
      int local_addr_length = 0;
      LPSOCKADDR remote_addr = 0;
      int remote_addr_length = 0;
      GetAcceptExSockaddrs(output_buffer, 0, address_length,
          address_length, &local_addr, &local_addr_length,
          &remote_addr, &remote_addr_length);
      if (static_cast<std::size_t>(remote_addr_length) > *addrlen)
      {
        ec = asio::error::invalid_argument;
      }
      else
      {
        using namespace std; // For memcpy.
        memcpy(addr, remote_addr, remote_addr_length);
        *addrlen = static_cast<std::size_t>(remote_addr_length);
      }
    }

    // Need to set the SO_UPDATE_ACCEPT_CONTEXT option so that getsockname
    // and getpeername will work on the accepted socket.
    SOCKET update_ctx_param = s;
    socket_ops::state_type state = 0;
    socket_ops::setsockopt(new_socket, state,
          SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
          &update_ctx_param, sizeof(SOCKET), ec);
  }
}

#else // defined(ASIO_HAS_IOCP)

bool non_blocking_accept(socket_type s,
    state_type state, socket_addr_type* addr, std::size_t* addrlen,
    asio::error_code& ec, socket_type& new_socket)
{
  for (;;)
  {
    // Accept the waiting connection.
    new_socket = socket_ops::accept(s, addr, addrlen, ec);

    // Check if operation succeeded.
    if (new_socket != invalid_socket)
      return true;

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Operation failed.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
    {
      // Fall through to retry operation.
    }
    else if (ec == asio::error::connection_aborted)
    {
      if (state & enable_connection_aborted)
        return true;
      // Fall through to retry operation.
    }
#if defined(EPROTO)
    else if (ec.value() == EPROTO)
    {
      if (state & enable_connection_aborted)
        return true;
      // Fall through to retry operation.
    }
#endif // defined(EPROTO)
    else
      return true;

    return false;
  }
}

#endif // defined(ASIO_HAS_IOCP)

template <typename SockLenType>
inline int call_bind(SockLenType msghdr::*,
    socket_type s, const socket_addr_type* addr, std::size_t addrlen)
{
  return ::bind(s, addr, (SockLenType)addrlen);
}

int bind(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
  int result = error_wrapper(call_bind(
        &msghdr::msg_namelen, s, addr, addrlen), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
}

int close(socket_type s, state_type& state,
    bool destruction, asio::error_code& ec)
{
  int result = 0;
  if (s != invalid_socket)
  {
    // We don't want the destructor to block, so set the socket to linger in
    // the background. If the user doesn't like this behaviour then they need
    // to explicitly close the socket.
    if (destruction && (state & user_set_linger))
    {
      ::linger opt;
      opt.l_onoff = 0;
      opt.l_linger = 0;
      asio::error_code ignored_ec;
      socket_ops::setsockopt(s, state, SOL_SOCKET,
          SO_LINGER, &opt, sizeof(opt), ignored_ec);
    }

    clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
    result = error_wrapper(::closesocket(s), ec);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
    result = error_wrapper(::close(s), ec);
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

    if (result != 0
        && (ec == asio::error::would_block
          || ec == asio::error::try_again))
    {
      // According to UNIX Network Programming Vol. 1, it is possible for
      // close() to fail with EWOULDBLOCK under certain circumstances. What
      // isn't clear is the state of the descriptor after this error. The one
      // current OS where this behaviour is seen, Windows, says that the socket
      // remains open. Therefore we'll put the descriptor back into blocking
      // mode and have another attempt at closing it.
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
      ioctl_arg_type arg = 0;
      ::ioctlsocket(s, FIONBIO, &arg);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# if defined(__SYMBIAN32__)
      int flags = ::fcntl(s, F_GETFL, 0);
      if (flags >= 0)
        ::fcntl(s, F_SETFL, flags & ~O_NONBLOCK);
# else // defined(__SYMBIAN32__)
      ioctl_arg_type arg = 0;
      ::ioctl(s, FIONBIO, &arg);
# endif // defined(__SYMBIAN32__)
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
      state &= ~non_blocking;

      clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
      result = error_wrapper(::closesocket(s), ec);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
      result = error_wrapper(::close(s), ec);
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
    }
  }

  if (result == 0)
    ec = asio::error_code();
  return result;
}

bool set_user_non_blocking(socket_type s,
    state_type& state, bool value, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return false;
  }

  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  ioctl_arg_type arg = (value ? 1 : 0);
  int result = error_wrapper(::ioctlsocket(s, FIONBIO, &arg), ec);
#elif defined(__SYMBIAN32__)
  int result = error_wrapper(::fcntl(s, F_GETFL, 0), ec);
  if (result >= 0)
  {
    clear_last_error();
    int flag = (value ? (result | O_NONBLOCK) : (result & ~O_NONBLOCK));
    result = error_wrapper(::fcntl(s, F_SETFL, flag), ec);
  }
#else
  ioctl_arg_type arg = (value ? 1 : 0);
  int result = error_wrapper(::ioctl(s, FIONBIO, &arg), ec);
#endif

  if (result >= 0)
  {
    ec = asio::error_code();
    if (value)
      state |= user_set_non_blocking;
    else
    {
      // Clearing the user-set non-blocking mode always overrides any
      // internally-set non-blocking flag. Any subsequent asynchronous
      // operations will need to re-enable non-blocking I/O.
      state &= ~(user_set_non_blocking | internal_non_blocking);
    }
    return true;
  }

  return false;
}

bool set_internal_non_blocking(socket_type s,
    state_type& state, bool value, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return false;
  }

  if (!value && (state & user_set_non_blocking))
  {
    // It does not make sense to clear the internal non-blocking flag if the
    // user still wants non-blocking behaviour. Return an error and let the
    // caller figure out whether to update the user-set non-blocking flag.
    ec = asio::error::invalid_argument;
    return false;
  }

  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  ioctl_arg_type arg = (value ? 1 : 0);
  int result = error_wrapper(::ioctlsocket(s, FIONBIO, &arg), ec);
#elif defined(__SYMBIAN32__)
  int result = error_wrapper(::fcntl(s, F_GETFL, 0), ec);
  if (result >= 0)
  {
    clear_last_error();
    int flag = (value ? (result | O_NONBLOCK) : (result & ~O_NONBLOCK));
    result = error_wrapper(::fcntl(s, F_SETFL, flag), ec);
  }
#else
  ioctl_arg_type arg = (value ? 1 : 0);
  int result = error_wrapper(::ioctl(s, FIONBIO, &arg), ec);
#endif

  if (result >= 0)
  {
    ec = asio::error_code();
    if (value)
      state |= internal_non_blocking;
    else
      state &= ~internal_non_blocking;
    return true;
  }

  return false;
}

int shutdown(socket_type s, int what, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
  int result = error_wrapper(::shutdown(s, what), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
}

template <typename SockLenType>
inline int call_connect(SockLenType msghdr::*,
    socket_type s, const socket_addr_type* addr, std::size_t addrlen)
{
  return ::connect(s, addr, (SockLenType)addrlen);
}

int connect(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
  int result = error_wrapper(call_connect(
        &msghdr::msg_namelen, s, addr, addrlen), ec);
  if (result == 0)
    ec = asio::error_code();
#if defined(__linux__)
  else if (ec == asio::error::try_again)
    ec = asio::error::no_buffer_space;
#endif // defined(__linux__)
  return result;
}

void sync_connect(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec)
{
  // Perform the connect operation.
  socket_ops::connect(s, addr, addrlen, ec);
  if (ec != asio::error::in_progress
      && ec != asio::error::would_block)
  {
    // The connect operation finished immediately.
    return;
  }

  // Wait for socket to become ready.
  if (socket_ops::poll_connect(s, -1, ec) < 0)
    return;

  // Get the error code from the connect operation.
  int connect_error = 0;
  size_t connect_error_len = sizeof(connect_error);
  if (socket_ops::getsockopt(s, 0, SOL_SOCKET, SO_ERROR,
        &connect_error, &connect_error_len, ec) == socket_error_retval)
    return;

  // Return the result of the connect operation.
  ec = asio::error_code(connect_error,
      asio::error::get_system_category());
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_connect(socket_type s, asio::error_code& ec)
{
  // Map non-portable errors to their portable counterparts.
  switch (ec.value())
  {
  case ERROR_CONNECTION_REFUSED:
    ec = asio::error::connection_refused;
    break;
  case ERROR_NETWORK_UNREACHABLE:
    ec = asio::error::network_unreachable;
    break;
  case ERROR_HOST_UNREACHABLE:
    ec = asio::error::host_unreachable;
    break;
  case ERROR_SEM_TIMEOUT:
    ec = asio::error::timed_out;
    break;
  default:
    break;
  }

  if (!ec)
  {
    // Need to set the SO_UPDATE_CONNECT_CONTEXT option so that getsockname
    // and getpeername will work on the connected socket.
    socket_ops::state_type state = 0;
    const int so_update_connect_context = 0x7010;
    socket_ops::setsockopt(s, state, SOL_SOCKET,
        so_update_connect_context, 0, 0, ec);
  }
}

#endif // defined(ASIO_HAS_IOCP)

bool non_blocking_connect(socket_type s, asio::error_code& ec)
{
  // Check if the connect operation has finished. This is required since we may
  // get spurious readiness notifications from the reactor.
#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)
  fd_set write_fds;
  FD_ZERO(&write_fds);
  FD_SET(s, &write_fds);
  fd_set except_fds;
  FD_ZERO(&except_fds);
  FD_SET(s, &except_fds);
  timeval zero_timeout;
  zero_timeout.tv_sec = 0;
  zero_timeout.tv_usec = 0;
  int ready = ::select(s + 1, 0, &write_fds, &except_fds, &zero_timeout);
#else // defined(ASIO_WINDOWS)
      // || defined(__CYGWIN__)
      // || defined(__SYMBIAN32__)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLOUT;
  fds.revents = 0;
  int ready = ::poll(&fds, 1, 0);
#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)
  if (ready == 0)
  {
    // The asynchronous connect operation is still in progress.
    return false;
  }

  // Get the error code from the connect operation.
  int connect_error = 0;
  size_t connect_error_len = sizeof(connect_error);
  if (socket_ops::getsockopt(s, 0, SOL_SOCKET, SO_ERROR,
        &connect_error, &connect_error_len, ec) == 0)
  {
    if (connect_error)
    {
      ec = asio::error_code(connect_error,
          asio::error::get_system_category());
    }
    else
      ec = asio::error_code();
  }

  return true;
}

int socketpair(int af, int type, int protocol,
    socket_type sv[2], asio::error_code& ec)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  (void)(af);
  (void)(type);
  (void)(protocol);
  (void)(sv);
  ec = asio::error::operation_not_supported;
  return socket_error_retval;
#else
  clear_last_error();
  int result = error_wrapper(::socketpair(af, type, protocol, sv), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
#endif
}

bool sockatmark(socket_type s, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return false;
  }

#if defined(SIOCATMARK)
  ioctl_arg_type value = 0;
# if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  int result = error_wrapper(::ioctlsocket(s, SIOCATMARK, &value), ec);
# else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  int result = error_wrapper(::ioctl(s, SIOCATMARK, &value), ec);
# endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  if (result == 0)
    ec = asio::error_code();
# if defined(ENOTTY)
  if (ec.value() == ENOTTY)
    ec = asio::error::not_socket;
# endif // defined(ENOTTY)
#else // defined(SIOCATMARK)
  int value = error_wrapper(::sockatmark(s), ec);
  if (value != -1)
    ec = asio::error_code();
#endif // defined(SIOCATMARK)

  return ec ? false : value != 0;
}

size_t available(socket_type s, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  ioctl_arg_type value = 0;
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  int result = error_wrapper(::ioctlsocket(s, FIONREAD, &value), ec);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  int result = error_wrapper(::ioctl(s, FIONREAD, &value), ec);
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  if (result == 0)
    ec = asio::error_code();
#if defined(ENOTTY)
  if (ec.value() == ENOTTY)
    ec = asio::error::not_socket;
#endif // defined(ENOTTY)

  return ec ? static_cast<size_t>(0) : static_cast<size_t>(value);
}

int listen(socket_type s, int backlog, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
  int result = error_wrapper(::listen(s, backlog), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
}

inline void init_buf_iov_base(void*& base, void* addr)
{
  base = addr;
}

template <typename T>
inline void init_buf_iov_base(T& base, void* addr)
{
  base = static_cast<T>(addr);
}

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef WSABUF buf;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef iovec buf;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

void init_buf(buf& b, void* data, size_t size)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  b.buf = static_cast<char*>(data);
  b.len = static_cast<u_long>(size);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  init_buf_iov_base(b.iov_base, data);
  b.iov_len = size;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

void init_buf(buf& b, const void* data, size_t size)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  b.buf = static_cast<char*>(const_cast<void*>(data));
  b.len = static_cast<u_long>(size);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  init_buf_iov_base(b.iov_base, const_cast<void*>(data));
  b.iov_len = size;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

inline void init_msghdr_msg_name(void*& name, socket_addr_type* addr)
{
  name = addr;
}

inline void init_msghdr_msg_name(void*& name, const socket_addr_type* addr)
{
  name = const_cast<socket_addr_type*>(addr);
}

template <typename T>
inline void init_msghdr_msg_name(T& name, socket_addr_type* addr)
{
  name = reinterpret_cast<T>(addr);
}

template <typename T>
inline void init_msghdr_msg_name(T& name, const socket_addr_type* addr)
{
  name = reinterpret_cast<T>(const_cast<socket_addr_type*>(addr));
}

signed_size_type recv(socket_type s, buf* bufs, size_t count,
    int flags, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  // Receive some data.
  DWORD recv_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD recv_flags = flags;
  int result = error_wrapper(::WSARecv(s, bufs,
        recv_buf_count, &bytes_transferred, &recv_flags, 0, 0), ec);
  if (ec.value() == ERROR_NETNAME_DELETED)
    ec = asio::error::connection_reset;
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
    ec = asio::error::connection_refused;
  else if (ec.value() == WSAEMSGSIZE || ec.value() == ERROR_MORE_DATA)
    ec.assign(0, ec.category());
  if (result != 0)
    return socket_error_retval;
  ec = asio::error_code();
  return bytes_transferred;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  msghdr msg = msghdr();
  msg.msg_iov = bufs;
  msg.msg_iovlen = static_cast<int>(count);
  signed_size_type result = error_wrapper(::recvmsg(s, &msg, flags), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

size_t sync_recv(socket_type s, state_type state, buf* bufs,
    size_t count, int flags, bool all_empty, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // A request to read 0 bytes on a stream is a no-op.
  if (all_empty && (state & stream_oriented))
  {
    ec = asio::error_code();
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::recv(s, bufs, count, flags, ec);

    // Check if operation succeeded.
    if (bytes > 0)
      return bytes;

    // Check for EOF.
    if ((state & stream_oriented) && bytes == 0)
    {
      ec = asio::error::eof;
      return 0;
    }

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_read(s, 0, -1, ec) < 0)
      return 0;
  }
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_recv(state_type state,
    const weak_cancel_token_type& cancel_token, bool all_empty,
    asio::error_code& ec, size_t bytes_transferred)
{
  // Map non-portable errors to their portable counterparts.
  if (ec.value() == ERROR_NETNAME_DELETED)
  {
    if (cancel_token.expired())
      ec = asio::error::operation_aborted;
    else
      ec = asio::error::connection_reset;
  }
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
  {
    ec = asio::error::connection_refused;
  }
  else if (ec.value() == WSAEMSGSIZE || ec.value() == ERROR_MORE_DATA)
  {
    ec.assign(0, ec.category());
  }

  // Check for connection closed.
  else if (!ec && bytes_transferred == 0
      && (state & stream_oriented) != 0
      && !all_empty)
  {
    ec = asio::error::eof;
  }
}

#else // defined(ASIO_HAS_IOCP)

bool non_blocking_recv(socket_type s,
    buf* bufs, size_t count, int flags, bool is_stream,
    asio::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Read some data.
    signed_size_type bytes = socket_ops::recv(s, bufs, count, flags, ec);

    // Check for end of stream.
    if (is_stream && bytes == 0)
    {
      ec = asio::error::eof;
      return true;
    }

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

#endif // defined(ASIO_HAS_IOCP)

signed_size_type recvfrom(socket_type s, buf* bufs, size_t count,
    int flags, socket_addr_type* addr, std::size_t* addrlen,
    asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  // Receive some data.
  DWORD recv_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD recv_flags = flags;
  int tmp_addrlen = (int)*addrlen;
  int result = error_wrapper(::WSARecvFrom(s, bufs, recv_buf_count,
        &bytes_transferred, &recv_flags, addr, &tmp_addrlen, 0, 0), ec);
  *addrlen = (std::size_t)tmp_addrlen;
  if (ec.value() == ERROR_NETNAME_DELETED)
    ec = asio::error::connection_reset;
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
    ec = asio::error::connection_refused;
  else if (ec.value() == WSAEMSGSIZE || ec.value() == ERROR_MORE_DATA)
    ec.assign(0, ec.category());
  if (result != 0)
    return socket_error_retval;
  ec = asio::error_code();
  return bytes_transferred;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  msghdr msg = msghdr();
  init_msghdr_msg_name(msg.msg_name, addr);
  msg.msg_namelen = static_cast<int>(*addrlen);
  msg.msg_iov = bufs;
  msg.msg_iovlen = static_cast<int>(count);
  signed_size_type result = error_wrapper(::recvmsg(s, &msg, flags), ec);
  *addrlen = msg.msg_namelen;
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

size_t sync_recvfrom(socket_type s, state_type state, buf* bufs,
    size_t count, int flags, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::recvfrom(
        s, bufs, count, flags, addr, addrlen, ec);

    // Check if operation succeeded.
    if (bytes >= 0)
      return bytes;

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_read(s, 0, -1, ec) < 0)
      return 0;
  }
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_recvfrom(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec)
{
  // Map non-portable errors to their portable counterparts.
  if (ec.value() == ERROR_NETNAME_DELETED)
  {
    if (cancel_token.expired())
      ec = asio::error::operation_aborted;
    else
      ec = asio::error::connection_reset;
  }
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
  {
    ec = asio::error::connection_refused;
  }
  else if (ec.value() == WSAEMSGSIZE || ec.value() == ERROR_MORE_DATA)
  {
    ec.assign(0, ec.category());
  }
}

#else // defined(ASIO_HAS_IOCP)

bool non_blocking_recvfrom(socket_type s,
    buf* bufs, size_t count, int flags,
    socket_addr_type* addr, std::size_t* addrlen,
    asio::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Read some data.
    signed_size_type bytes = socket_ops::recvfrom(
        s, bufs, count, flags, addr, addrlen, ec);

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

#endif // defined(ASIO_HAS_IOCP)

signed_size_type recvmsg(socket_type s, buf* bufs, size_t count,
    int in_flags, int& out_flags, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  out_flags = 0;
  return socket_ops::recv(s, bufs, count, in_flags, ec);
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  msghdr msg = msghdr();
  msg.msg_iov = bufs;
  msg.msg_iovlen = static_cast<int>(count);
  signed_size_type result = error_wrapper(::recvmsg(s, &msg, in_flags), ec);
  if (result >= 0)
  {
    ec = asio::error_code();
    out_flags = msg.msg_flags;
  }
  else
    out_flags = 0;
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

size_t sync_recvmsg(socket_type s, state_type state,
    buf* bufs, size_t count, int in_flags, int& out_flags,
    asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::recvmsg(
        s, bufs, count, in_flags, out_flags, ec);

    // Check if operation succeeded.
    if (bytes >= 0)
      return bytes;

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_read(s, 0, -1, ec) < 0)
      return 0;
  }
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_recvmsg(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec)
{
  // Map non-portable errors to their portable counterparts.
  if (ec.value() == ERROR_NETNAME_DELETED)
  {
    if (cancel_token.expired())
      ec = asio::error::operation_aborted;
    else
      ec = asio::error::connection_reset;
  }
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
  {
    ec = asio::error::connection_refused;
  }
  else if (ec.value() == WSAEMSGSIZE || ec.value() == ERROR_MORE_DATA)
  {
    ec.assign(0, ec.category());
  }
}

#else // defined(ASIO_HAS_IOCP)

bool non_blocking_recvmsg(socket_type s,
    buf* bufs, size_t count, int in_flags, int& out_flags,
    asio::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Read some data.
    signed_size_type bytes = socket_ops::recvmsg(
        s, bufs, count, in_flags, out_flags, ec);

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

#endif // defined(ASIO_HAS_IOCP)

signed_size_type send(socket_type s, const buf* bufs, size_t count,
    int flags, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  // Send the data.
  DWORD send_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD send_flags = flags;
  int result = error_wrapper(::WSASend(s, const_cast<buf*>(bufs),
        send_buf_count, &bytes_transferred, send_flags, 0, 0), ec);
  if (ec.value() == ERROR_NETNAME_DELETED)
    ec = asio::error::connection_reset;
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
    ec = asio::error::connection_refused;
  if (result != 0)
    return socket_error_retval;
  ec = asio::error_code();
  return bytes_transferred;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  msghdr msg = msghdr();
  msg.msg_iov = const_cast<buf*>(bufs);
  msg.msg_iovlen = static_cast<int>(count);
#if defined(__linux__)
  flags |= MSG_NOSIGNAL;
#endif // defined(__linux__)
  signed_size_type result = error_wrapper(::sendmsg(s, &msg, flags), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

size_t sync_send(socket_type s, state_type state, const buf* bufs,
    size_t count, int flags, bool all_empty, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // A request to write 0 bytes to a stream is a no-op.
  if (all_empty && (state & stream_oriented))
  {
    ec = asio::error_code();
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::send(s, bufs, count, flags, ec);

    // Check if operation succeeded.
    if (bytes >= 0)
      return bytes;

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_write(s, 0, -1, ec) < 0)
      return 0;
  }
}

#if defined(ASIO_HAS_IOCP)

void complete_iocp_send(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec)
{
  // Map non-portable errors to their portable counterparts.
  if (ec.value() == ERROR_NETNAME_DELETED)
  {
    if (cancel_token.expired())
      ec = asio::error::operation_aborted;
    else
      ec = asio::error::connection_reset;
  }
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
  {
    ec = asio::error::connection_refused;
  }
}

#else // defined(ASIO_HAS_IOCP)

bool non_blocking_send(socket_type s,
    const buf* bufs, size_t count, int flags,
    asio::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Write some data.
    signed_size_type bytes = socket_ops::send(s, bufs, count, flags, ec);

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

#endif // defined(ASIO_HAS_IOCP)

signed_size_type sendto(socket_type s, const buf* bufs, size_t count,
    int flags, const socket_addr_type* addr, std::size_t addrlen,
    asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  // Send the data.
  DWORD send_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  int result = error_wrapper(::WSASendTo(s, const_cast<buf*>(bufs),
        send_buf_count, &bytes_transferred, flags, addr,
        static_cast<int>(addrlen), 0, 0), ec);
  if (ec.value() == ERROR_NETNAME_DELETED)
    ec = asio::error::connection_reset;
  else if (ec.value() == ERROR_PORT_UNREACHABLE)
    ec = asio::error::connection_refused;
  if (result != 0)
    return socket_error_retval;
  ec = asio::error_code();
  return bytes_transferred;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  msghdr msg = msghdr();
  init_msghdr_msg_name(msg.msg_name, addr);
  msg.msg_namelen = static_cast<int>(addrlen);
  msg.msg_iov = const_cast<buf*>(bufs);
  msg.msg_iovlen = static_cast<int>(count);
#if defined(__linux__)
  flags |= MSG_NOSIGNAL;
#endif // defined(__linux__)
  signed_size_type result = error_wrapper(::sendmsg(s, &msg, flags), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

size_t sync_sendto(socket_type s, state_type state, const buf* bufs,
    size_t count, int flags, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // Write some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::sendto(
        s, bufs, count, flags, addr, addrlen, ec);

    // Check if operation succeeded.
    if (bytes >= 0)
      return bytes;

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_write(s, 0, -1, ec) < 0)
      return 0;
  }
}

#if !defined(ASIO_HAS_IOCP)

bool non_blocking_sendto(socket_type s,
    const buf* bufs, size_t count, int flags,
    const socket_addr_type* addr, std::size_t addrlen,
    asio::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Write some data.
    signed_size_type bytes = socket_ops::sendto(
        s, bufs, count, flags, addr, addrlen, ec);

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

#endif // !defined(ASIO_HAS_IOCP)

socket_type socket(int af, int type, int protocol,
    asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  socket_type s = error_wrapper(::WSASocketW(af, type, protocol, 0, 0,
        WSA_FLAG_OVERLAPPED), ec);
  if (s == invalid_socket)
    return s;

  if (af == ASIO_OS_DEF(AF_INET6))
  {
    // Try to enable the POSIX default behaviour of having IPV6_V6ONLY set to
    // false. This will only succeed on Windows Vista and later versions of
    // Windows, where a dual-stack IPv4/v6 implementation is available.
    DWORD optval = 0;
    ::setsockopt(s, IPPROTO_IPV6, IPV6_V6ONLY,
        reinterpret_cast<const char*>(&optval), sizeof(optval));
  }

  ec = asio::error_code();

  return s;
#elif defined(__MACH__) && defined(__APPLE__) || defined(__FreeBSD__)
  socket_type s = error_wrapper(::socket(af, type, protocol), ec);
  if (s == invalid_socket)
    return s;

  int optval = 1;
  int result = error_wrapper(::setsockopt(s,
        SOL_SOCKET, SO_NOSIGPIPE, &optval, sizeof(optval)), ec);
  if (result != 0)
  {
    ::close(s);
    return invalid_socket;
  }

  return s;
#else
  int s = error_wrapper(::socket(af, type, protocol), ec);
  if (s >= 0)
    ec = asio::error_code();
  return s;
#endif
}

template <typename SockLenType>
inline int call_setsockopt(SockLenType msghdr::*,
    socket_type s, int level, int optname,
    const void* optval, std::size_t optlen)
{
  return ::setsockopt(s, level, optname,
      (const char*)optval, (SockLenType)optlen);
}

int setsockopt(socket_type s, state_type& state, int level, int optname,
    const void* optval, std::size_t optlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  if (level == custom_socket_option_level && optname == always_fail_option)
  {
    ec = asio::error::invalid_argument;
    return socket_error_retval;
  }

  if (level == custom_socket_option_level
      && optname == enable_connection_aborted_option)
  {
    if (optlen != sizeof(int))
    {
      ec = asio::error::invalid_argument;
      return socket_error_retval;
    }

    if (*static_cast<const int*>(optval))
      state |= enable_connection_aborted;
    else
      state &= ~enable_connection_aborted;
    ec = asio::error_code();
    return 0;
  }

  if (level == SOL_SOCKET && optname == SO_LINGER)
    state |= user_set_linger;

#if defined(__BORLANDC__)
  // Mysteriously, using the getsockopt and setsockopt functions directly with
  // Borland C++ results in incorrect values being set and read. The bug can be
  // worked around by using function addresses resolved with GetProcAddress.
  if (HMODULE winsock_module = ::GetModuleHandleA("ws2_32"))
  {
    typedef int (WSAAPI *sso_t)(SOCKET, int, int, const char*, int);
    if (sso_t sso = (sso_t)::GetProcAddress(winsock_module, "setsockopt"))
    {
      clear_last_error();
      return error_wrapper(sso(s, level, optname,
            reinterpret_cast<const char*>(optval),
            static_cast<int>(optlen)), ec);
    }
  }
  ec = asio::error::fault;
  return socket_error_retval;
#else // defined(__BORLANDC__)
  clear_last_error();
  int result = error_wrapper(call_setsockopt(&msghdr::msg_namelen,
        s, level, optname, optval, optlen), ec);
  if (result == 0)
  {
    ec = asio::error_code();

#if defined(__MACH__) && defined(__APPLE__) \
  || defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    // To implement portable behaviour for SO_REUSEADDR with UDP sockets we
    // need to also set SO_REUSEPORT on BSD-based platforms.
    if ((state & datagram_oriented)
        && level == SOL_SOCKET && optname == SO_REUSEADDR)
    {
      call_setsockopt(&msghdr::msg_namelen, s,
          SOL_SOCKET, SO_REUSEPORT, optval, optlen);
    }
#endif
  }

  return result;
#endif // defined(__BORLANDC__)
}

template <typename SockLenType>
inline int call_getsockopt(SockLenType msghdr::*,
    socket_type s, int level, int optname,
    void* optval, std::size_t* optlen)
{
  SockLenType tmp_optlen = (SockLenType)*optlen;
  int result = ::getsockopt(s, level, optname, (char*)optval, &tmp_optlen);
  *optlen = (std::size_t)tmp_optlen;
  return result;
}

int getsockopt(socket_type s, state_type state, int level, int optname,
    void* optval, size_t* optlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  if (level == custom_socket_option_level && optname == always_fail_option)
  {
    ec = asio::error::invalid_argument;
    return socket_error_retval;
  }

  if (level == custom_socket_option_level
      && optname == enable_connection_aborted_option)
  {
    if (*optlen != sizeof(int))
    {
      ec = asio::error::invalid_argument;
      return socket_error_retval;
    }

    *static_cast<int*>(optval) = (state & enable_connection_aborted) ? 1 : 0;
    ec = asio::error_code();
    return 0;
  }

#if defined(__BORLANDC__)
  // Mysteriously, using the getsockopt and setsockopt functions directly with
  // Borland C++ results in incorrect values being set and read. The bug can be
  // worked around by using function addresses resolved with GetProcAddress.
  if (HMODULE winsock_module = ::GetModuleHandleA("ws2_32"))
  {
    typedef int (WSAAPI *gso_t)(SOCKET, int, int, char*, int*);
    if (gso_t gso = (gso_t)::GetProcAddress(winsock_module, "getsockopt"))
    {
      clear_last_error();
      int tmp_optlen = static_cast<int>(*optlen);
      int result = error_wrapper(gso(s, level, optname,
            reinterpret_cast<char*>(optval), &tmp_optlen), ec);
      *optlen = static_cast<size_t>(tmp_optlen);
      if (result != 0 && level == IPPROTO_IPV6 && optname == IPV6_V6ONLY
          && ec.value() == WSAENOPROTOOPT && *optlen == sizeof(DWORD))
      {
        // Dual-stack IPv4/v6 sockets, and the IPV6_V6ONLY socket option, are
        // only supported on Windows Vista and later. To simplify program logic
        // we will fake success of getting this option and specify that the
        // value is non-zero (i.e. true). This corresponds to the behavior of
        // IPv6 sockets on Windows platforms pre-Vista.
        *static_cast<DWORD*>(optval) = 1;
        ec = asio::error_code();
      }
      return result;
    }
  }
  ec = asio::error::fault;
  return socket_error_retval;
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  clear_last_error();
  int result = error_wrapper(call_getsockopt(&msghdr::msg_namelen,
        s, level, optname, optval, optlen), ec);
  if (result != 0 && level == IPPROTO_IPV6 && optname == IPV6_V6ONLY
      && ec.value() == WSAENOPROTOOPT && *optlen == sizeof(DWORD))
  {
    // Dual-stack IPv4/v6 sockets, and the IPV6_V6ONLY socket option, are only
    // supported on Windows Vista and later. To simplify program logic we will
    // fake success of getting this option and specify that the value is
    // non-zero (i.e. true). This corresponds to the behavior of IPv6 sockets
    // on Windows platforms pre-Vista.
    *static_cast<DWORD*>(optval) = 1;
    ec = asio::error_code();
  }
  if (result == 0)
    ec = asio::error_code();
  return result;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  clear_last_error();
  int result = error_wrapper(call_getsockopt(&msghdr::msg_namelen,
        s, level, optname, optval, optlen), ec);
#if defined(__linux__)
  if (result == 0 && level == SOL_SOCKET && *optlen == sizeof(int)
      && (optname == SO_SNDBUF || optname == SO_RCVBUF))
  {
    // On Linux, setting SO_SNDBUF or SO_RCVBUF to N actually causes the kernel
    // to set the buffer size to N*2. Linux puts additional stuff into the
    // buffers so that only about half is actually available to the application.
    // The retrieved value is divided by 2 here to make it appear as though the
    // correct value has been set.
    *static_cast<int*>(optval) /= 2;
  }
#endif // defined(__linux__)
  if (result == 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

template <typename SockLenType>
inline int call_getpeername(SockLenType msghdr::*,
    socket_type s, socket_addr_type* addr, std::size_t* addrlen)
{
  SockLenType tmp_addrlen = (SockLenType)*addrlen;
  int result = ::getpeername(s, addr, &tmp_addrlen);
  *addrlen = (std::size_t)tmp_addrlen;
  return result;
}

int getpeername(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, bool cached, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

#if defined(ASIO_WINDOWS) && !defined(ASIO_WINDOWS_APP) \
  || defined(__CYGWIN__)
  if (cached)
  {
    // Check if socket is still connected.
    DWORD connect_time = 0;
    size_t connect_time_len = sizeof(connect_time);
    if (socket_ops::getsockopt(s, 0, SOL_SOCKET, SO_CONNECT_TIME,
          &connect_time, &connect_time_len, ec) == socket_error_retval)
    {
      return socket_error_retval;
    }
    if (connect_time == 0xFFFFFFFF)
    {
      ec = asio::error::not_connected;
      return socket_error_retval;
    }

    // The cached value is still valid.
    ec = asio::error_code();
    return 0;
  }
#else // defined(ASIO_WINDOWS) && !defined(ASIO_WINDOWS_APP)
      // || defined(__CYGWIN__)
  (void)cached;
#endif // defined(ASIO_WINDOWS) && !defined(ASIO_WINDOWS_APP)
       // || defined(__CYGWIN__)

  clear_last_error();
  int result = error_wrapper(call_getpeername(
        &msghdr::msg_namelen, s, addr, addrlen), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
}

template <typename SockLenType>
inline int call_getsockname(SockLenType msghdr::*,
    socket_type s, socket_addr_type* addr, std::size_t* addrlen)
{
  SockLenType tmp_addrlen = (SockLenType)*addrlen;
  int result = ::getsockname(s, addr, &tmp_addrlen);
  *addrlen = (std::size_t)tmp_addrlen;
  return result;
}

int getsockname(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
  int result = error_wrapper(call_getsockname(
        &msghdr::msg_namelen, s, addr, addrlen), ec);
  if (result == 0)
    ec = asio::error_code();
  return result;
}

int ioctl(socket_type s, state_type& state, int cmd,
    ioctl_arg_type* arg, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  int result = error_wrapper(::ioctlsocket(s, cmd, arg), ec);
#elif defined(__MACH__) && defined(__APPLE__) \
  || defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__)
  int result = error_wrapper(::ioctl(s,
        static_cast<unsigned int>(cmd), arg), ec);
#else
  int result = error_wrapper(::ioctl(s, cmd, arg), ec);
#endif
  if (result >= 0)
  {
    ec = asio::error_code();

    // When updating the non-blocking mode we always perform the ioctl syscall,
    // even if the flags would otherwise indicate that the socket is already in
    // the correct state. This ensures that the underlying socket is put into
    // the state that has been requested by the user. If the ioctl syscall was
    // successful then we need to update the flags to match.
    if (cmd == static_cast<int>(FIONBIO))
    {
      if (*arg)
      {
        state |= user_set_non_blocking;
      }
      else
      {
        // Clearing the non-blocking mode always overrides any internally-set
        // non-blocking flag. Any subsequent asynchronous operations will need
        // to re-enable non-blocking I/O.
        state &= ~(user_set_non_blocking | internal_non_blocking);
      }
    }
  }

  return result;
}

int select(int nfds, fd_set* readfds, fd_set* writefds,
    fd_set* exceptfds, timeval* timeout, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  if (!readfds && !writefds && !exceptfds && timeout)
  {
    DWORD milliseconds = timeout->tv_sec * 1000 + timeout->tv_usec / 1000;
    if (milliseconds == 0)
      milliseconds = 1; // Force context switch.
    ::Sleep(milliseconds);
    ec = asio::error_code();
    return 0;
  }

  // The select() call allows timeout values measured in microseconds, but the
  // system clock (as wrapped by boost::posix_time::microsec_clock) typically
  // has a resolution of 10 milliseconds. This can lead to a spinning select
  // reactor, meaning increased CPU usage, when waiting for the earliest
  // scheduled timeout if it's less than 10 milliseconds away. To avoid a tight
  // spin we'll use a minimum timeout of 1 millisecond.
  if (timeout && timeout->tv_sec == 0
      && timeout->tv_usec > 0 && timeout->tv_usec < 1000)
    timeout->tv_usec = 1000;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

#if defined(__hpux) && defined(__SELECT)
  timespec ts;
  ts.tv_sec = timeout ? timeout->tv_sec : 0;
  ts.tv_nsec = timeout ? timeout->tv_usec * 1000 : 0;
  return error_wrapper(::pselect(nfds, readfds,
        writefds, exceptfds, timeout ? &ts : 0, 0), ec);
#else
  int result = error_wrapper(::select(nfds, readfds,
        writefds, exceptfds, timeout), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif
}

int poll_read(socket_type s, state_type state,
    int msec, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  timeval timeout_obj;
  timeval* timeout;
  if (state & user_set_non_blocking)
  {
    timeout_obj.tv_sec = 0;
    timeout_obj.tv_usec = 0;
    timeout = &timeout_obj;
  }
  else if (msec >= 0)
  {
    timeout_obj.tv_sec = msec / 1000;
    timeout_obj.tv_usec = (msec % 1000) * 1000;
    timeout = &timeout_obj;
  }
  else
    timeout = 0;
  clear_last_error();
  int result = error_wrapper(::select(s + 1, &fds, 0, 0, timeout), ec);
#else // defined(ASIO_WINDOWS)
      // || defined(__CYGWIN__)
      // || defined(__SYMBIAN32__)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLIN;
  fds.revents = 0;
  int timeout = (state & user_set_non_blocking) ? 0 : msec;
  clear_last_error();
  int result = error_wrapper(::poll(&fds, 1, timeout), ec);
#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)
  if (result == 0)
    ec = (state & user_set_non_blocking)
      ? asio::error::would_block : asio::error_code();
  else if (result > 0)
    ec = asio::error_code();
  return result;
}

int poll_write(socket_type s, state_type state,
    int msec, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  timeval timeout_obj;
  timeval* timeout;
  if (state & user_set_non_blocking)
  {
    timeout_obj.tv_sec = 0;
    timeout_obj.tv_usec = 0;
    timeout = &timeout_obj;
  }
  else if (msec >= 0)
  {
    timeout_obj.tv_sec = msec / 1000;
    timeout_obj.tv_usec = (msec % 1000) * 1000;
    timeout = &timeout_obj;
  }
  else
    timeout = 0;
  clear_last_error();
  int result = error_wrapper(::select(s + 1, 0, &fds, 0, timeout), ec);
#else // defined(ASIO_WINDOWS)
      // || defined(__CYGWIN__)
      // || defined(__SYMBIAN32__)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLOUT;
  fds.revents = 0;
  int timeout = (state & user_set_non_blocking) ? 0 : msec;
  clear_last_error();
  int result = error_wrapper(::poll(&fds, 1, timeout), ec);
#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)
  if (result == 0)
    ec = (state & user_set_non_blocking)
      ? asio::error::would_block : asio::error_code();
  else if (result > 0)
    ec = asio::error_code();
  return result;
}

int poll_error(socket_type s, state_type state,
    int msec, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  timeval timeout_obj;
  timeval* timeout;
  if (state & user_set_non_blocking)
  {
    timeout_obj.tv_sec = 0;
    timeout_obj.tv_usec = 0;
    timeout = &timeout_obj;
  }
  else if (msec >= 0)
  {
    timeout_obj.tv_sec = msec / 1000;
    timeout_obj.tv_usec = (msec % 1000) * 1000;
    timeout = &timeout_obj;
  }
  else
    timeout = 0;
  clear_last_error();
  int result = error_wrapper(::select(s + 1, 0, 0, &fds, timeout), ec);
#else // defined(ASIO_WINDOWS)
      // || defined(__CYGWIN__)
      // || defined(__SYMBIAN32__)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLPRI | POLLERR | POLLHUP;
  fds.revents = 0;
  int timeout = (state & user_set_non_blocking) ? 0 : msec;
  clear_last_error();
  int result = error_wrapper(::poll(&fds, 1, timeout), ec);
#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)
  if (result == 0)
    ec = (state & user_set_non_blocking)
      ? asio::error::would_block : asio::error_code();
  else if (result > 0)
    ec = asio::error_code();
  return result;
}

int poll_connect(socket_type s, int msec, asio::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = asio::error::bad_descriptor;
    return socket_error_retval;
  }

#if defined(ASIO_WINDOWS) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)
  fd_set write_fds;
  FD_ZERO(&write_fds);
  FD_SET(s, &write_fds);
  fd_set except_fds;
  FD_ZERO(&except_fds);
  FD_SET(s, &except_fds);
  timeval timeout_obj;
  timeval* timeout;
  if (msec >= 0)
  {
    timeout_obj.tv_sec = msec / 1000;
    timeout_obj.tv_usec = (msec % 1000) * 1000;
    timeout = &timeout_obj;
  }
  else
    timeout = 0;
  clear_last_error();
  int result = error_wrapper(::select(
        s + 1, 0, &write_fds, &except_fds, timeout), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#else // defined(ASIO_WINDOWS)
      // || defined(__CYGWIN__)
      // || defined(__SYMBIAN32__)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLOUT;
  fds.revents = 0;
  clear_last_error();
  int result = error_wrapper(::poll(&fds, 1, msec), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
#endif // defined(ASIO_WINDOWS)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)
}

#endif // !defined(ASIO_WINDOWS_RUNTIME)

const char* inet_ntop(int af, const void* src, char* dest, size_t length,
    unsigned long scope_id, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS_RUNTIME)
  using namespace std; // For sprintf.
  const unsigned char* bytes = static_cast<const unsigned char*>(src);
  if (af == ASIO_OS_DEF(AF_INET))
  {
    sprintf_s(dest, length, "%u.%u.%u.%u",
        bytes[0], bytes[1], bytes[2], bytes[3]);
    return dest;
  }
  else if (af == ASIO_OS_DEF(AF_INET6))
  {
    size_t n = 0, b = 0, z = 0;
    while (n < length && b < 16)
    {
      if (bytes[b] == 0 && bytes[b + 1] == 0 && z == 0)
      {
        do b += 2; while (b < 16 && bytes[b] == 0 && bytes[b + 1] == 0);
        n += sprintf_s(dest + n, length - n, ":%s", b < 16 ? "" : ":"), ++z;
      }
      else
      {
        n += sprintf_s(dest + n, length - n, "%s%x", b ? ":" : "",
            (static_cast<u_long_type>(bytes[b]) << 8) | bytes[b + 1]);
        b += 2;
      }
    }
    if (scope_id)
      n += sprintf_s(dest + n, length - n, "%%%lu", scope_id);
    return dest;
  }
  else
  {
    ec = asio::error::address_family_not_supported;
    return 0;
  }
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  using namespace std; // For memcpy.

  if (af != ASIO_OS_DEF(AF_INET) && af != ASIO_OS_DEF(AF_INET6))
  {
    ec = asio::error::address_family_not_supported;
    return 0;
  }

  union
  {
    socket_addr_type base;
    sockaddr_storage_type storage;
    sockaddr_in4_type v4;
    sockaddr_in6_type v6;
  } address;
  DWORD address_length;
  if (af == ASIO_OS_DEF(AF_INET))
  {
    address_length = sizeof(sockaddr_in4_type);
    address.v4.sin_family = ASIO_OS_DEF(AF_INET);
    address.v4.sin_port = 0;
    memcpy(&address.v4.sin_addr, src, sizeof(in4_addr_type));
  }
  else // AF_INET6
  {
    address_length = sizeof(sockaddr_in6_type);
    address.v6.sin6_family = ASIO_OS_DEF(AF_INET6);
    address.v6.sin6_port = 0;
    address.v6.sin6_flowinfo = 0;
    address.v6.sin6_scope_id = scope_id;
    memcpy(&address.v6.sin6_addr, src, sizeof(in6_addr_type));
  }

  DWORD string_length = static_cast<DWORD>(length);
#if defined(BOOST_NO_ANSI_APIS) || (defined(_MSC_VER) && (_MSC_VER >= 1800))
  LPWSTR string_buffer = (LPWSTR)_alloca(length * sizeof(WCHAR));
  int result = error_wrapper(::WSAAddressToStringW(&address.base,
        address_length, 0, string_buffer, &string_length), ec);
  ::WideCharToMultiByte(CP_ACP, 0, string_buffer, -1,
      dest, static_cast<int>(length), 0, 0);
#else
  int result = error_wrapper(::WSAAddressToStringA(
        &address.base, address_length, 0, dest, &string_length), ec);
#endif

  // Windows may set error code on success.
  if (result != socket_error_retval)
    ec = asio::error_code();

  // Windows may not set an error code on failure.
  else if (result == socket_error_retval && !ec)
    ec = asio::error::invalid_argument;

  return result == socket_error_retval ? 0 : dest;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  const char* result = error_wrapper(::inet_ntop(
        af, src, dest, static_cast<int>(length)), ec);
  if (result == 0 && !ec)
    ec = asio::error::invalid_argument;
  if (result != 0 && af == ASIO_OS_DEF(AF_INET6) && scope_id != 0)
  {
    using namespace std; // For strcat and sprintf.
    char if_name[IF_NAMESIZE + 1] = "%";
    const in6_addr_type* ipv6_address = static_cast<const in6_addr_type*>(src);
    bool is_link_local = ((ipv6_address->s6_addr[0] == 0xfe)
        && ((ipv6_address->s6_addr[1] & 0xc0) == 0x80));
    bool is_multicast_link_local = ((ipv6_address->s6_addr[0] == 0xff)
        && ((ipv6_address->s6_addr[1] & 0x0f) == 0x02));
    if ((!is_link_local && !is_multicast_link_local)
        || if_indextoname(static_cast<unsigned>(scope_id), if_name + 1) == 0)
      sprintf(if_name + 1, "%lu", scope_id);
    strcat(dest, if_name);
  }
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

int inet_pton(int af, const char* src, void* dest,
    unsigned long* scope_id, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS_RUNTIME)
  using namespace std; // For sscanf.
  unsigned char* bytes = static_cast<unsigned char*>(dest);
  if (af == ASIO_OS_DEF(AF_INET))
  {
    unsigned int b0, b1, b2, b3;
    if (sscanf_s(src, "%u.%u.%u.%u", &b0, &b1, &b2, &b3) != 4)
    {
      ec = asio::error::invalid_argument;
      return -1;
    }
    if (b0 > 255 || b1 > 255 || b2 > 255 || b3 > 255)
    {
      ec = asio::error::invalid_argument;
      return -1;
    }
    bytes[0] = static_cast<unsigned char>(b0);
    bytes[1] = static_cast<unsigned char>(b1);
    bytes[2] = static_cast<unsigned char>(b2);
    bytes[3] = static_cast<unsigned char>(b3);
    ec = asio::error_code();
    return 1;
  }
  else if (af == ASIO_OS_DEF(AF_INET6))
  {
    unsigned char* bytes = static_cast<unsigned char*>(dest);
    std::memset(bytes, 0, 16);
    unsigned char back_bytes[16] = { 0 };
    int num_front_bytes = 0, num_back_bytes = 0;
    const char* p = src;

    enum { fword, fcolon, bword, scope, done } state = fword;
    unsigned long current_word = 0;
    while (state != done)
    {
      if (current_word > 0xFFFF)
      {
        ec = asio::error::invalid_argument;
        return -1;
      }

      switch (state)
      {
      case fword:
        if (*p >= '0' && *p <= '9')
          current_word = current_word * 16 + *p++ - '0';
        else if (*p >= 'a' && *p <= 'f')
          current_word = current_word * 16 + *p++ - 'a' + 10;
        else if (*p >= 'A' && *p <= 'F')
          current_word = current_word * 16 + *p++ - 'A' + 10;
        else
        {
          if (num_front_bytes == 16)
          {
            ec = asio::error::invalid_argument;
            return -1;
          }

          bytes[num_front_bytes++] = (current_word >> 8) & 0xFF;
          bytes[num_front_bytes++] = current_word & 0xFF;
          current_word = 0;

          if (*p == ':')
            state = fcolon, ++p;
          else if (*p == '%')
            state = scope, ++p;
          else if (*p == 0)
            state = done;
          else
          {
            ec = asio::error::invalid_argument;
            return -1;
          }
        }
        break;

      case fcolon:
        if (*p == ':')
          state = bword, ++p;
        else
          state = fword;
        break;

      case bword:
        if (*p >= '0' && *p <= '9')
          current_word = current_word * 16 + *p++ - '0';
        else if (*p >= 'a' && *p <= 'f')
          current_word = current_word * 16 + *p++ - 'a' + 10;
        else if (*p >= 'A' && *p <= 'F')
          current_word = current_word * 16 + *p++ - 'A' + 10;
        else
        {
          if (num_front_bytes + num_back_bytes == 16)
          {
            ec = asio::error::invalid_argument;
            return -1;
          }

          back_bytes[num_back_bytes++] = (current_word >> 8) & 0xFF;
          back_bytes[num_back_bytes++] = current_word & 0xFF;
          current_word = 0;

          if (*p == ':')
            state = bword, ++p;
          else if (*p == '%')
            state = scope, ++p;
          else if (*p == 0)
            state = done;
          else
          {
            ec = asio::error::invalid_argument;
            return -1;
          }
        }
        break;

      case scope:
        if (*p >= '0' && *p <= '9')
          current_word = current_word * 10 + *p++ - '0';
        else if (*p == 0)
          *scope_id = current_word, state = done;
        else
        {
          ec = asio::error::invalid_argument;
          return -1;
        }
        break;

      default:
        break;
      }
    }

    for (int i = 0; i < num_back_bytes; ++i)
      bytes[16 - num_back_bytes + i] = back_bytes[i];

    ec = asio::error_code();
    return 1;
  }
  else
  {
    ec = asio::error::address_family_not_supported;
    return -1;
  }
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  using namespace std; // For memcpy and strcmp.

  if (af != ASIO_OS_DEF(AF_INET) && af != ASIO_OS_DEF(AF_INET6))
  {
    ec = asio::error::address_family_not_supported;
    return -1;
  }

  union
  {
    socket_addr_type base;
    sockaddr_storage_type storage;
    sockaddr_in4_type v4;
    sockaddr_in6_type v6;
  } address;
  int address_length = sizeof(sockaddr_storage_type);
#if defined(BOOST_NO_ANSI_APIS) || (defined(_MSC_VER) && (_MSC_VER >= 1800))
  int num_wide_chars = static_cast<int>(strlen(src)) + 1;
  LPWSTR wide_buffer = (LPWSTR)_alloca(num_wide_chars * sizeof(WCHAR));
  ::MultiByteToWideChar(CP_ACP, 0, src, -1, wide_buffer, num_wide_chars);
  int result = error_wrapper(::WSAStringToAddressW(
        wide_buffer, af, 0, &address.base, &address_length), ec);
#else
  int result = error_wrapper(::WSAStringToAddressA(
        const_cast<char*>(src), af, 0, &address.base, &address_length), ec);
#endif

  if (af == ASIO_OS_DEF(AF_INET))
  {
    if (result != socket_error_retval)
    {
      memcpy(dest, &address.v4.sin_addr, sizeof(in4_addr_type));
      ec = asio::error_code();
    }
    else if (strcmp(src, "255.255.255.255") == 0)
    {
      static_cast<in4_addr_type*>(dest)->s_addr = INADDR_NONE;
      ec = asio::error_code();
    }
  }
  else // AF_INET6
  {
    if (result != socket_error_retval)
    {
      memcpy(dest, &address.v6.sin6_addr, sizeof(in6_addr_type));
      if (scope_id)
        *scope_id = address.v6.sin6_scope_id;
      ec = asio::error_code();
    }
  }

  // Windows may not set an error code on failure.
  if (result == socket_error_retval && !ec)
    ec = asio::error::invalid_argument;

  if (result != socket_error_retval)
    ec = asio::error_code();

  return result == socket_error_retval ? -1 : 1;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  using namespace std; // For strchr, memcpy and atoi.

  // On some platforms, inet_pton fails if an address string contains a scope
  // id. Detect and remove the scope id before passing the string to inet_pton.
  const bool is_v6 = (af == ASIO_OS_DEF(AF_INET6));
  const char* if_name = is_v6 ? strchr(src, '%') : 0;
  char src_buf[max_addr_v6_str_len + 1];
  const char* src_ptr = src;
  if (if_name != 0)
  {
    if (if_name - src > max_addr_v6_str_len)
    {
      ec = asio::error::invalid_argument;
      return 0;
    }
    memcpy(src_buf, src, if_name - src);
    src_buf[if_name - src] = 0;
    src_ptr = src_buf;
  }

  int result = error_wrapper(::inet_pton(af, src_ptr, dest), ec);
  if (result <= 0 && !ec)
    ec = asio::error::invalid_argument;
  if (result > 0 && is_v6 && scope_id)
  {
    using namespace std; // For strchr and atoi.
    *scope_id = 0;
    if (if_name != 0)
    {
      in6_addr_type* ipv6_address = static_cast<in6_addr_type*>(dest);
      bool is_link_local = ((ipv6_address->s6_addr[0] == 0xfe)
          && ((ipv6_address->s6_addr[1] & 0xc0) == 0x80));
      bool is_multicast_link_local = ((ipv6_address->s6_addr[0] == 0xff)
          && ((ipv6_address->s6_addr[1] & 0x0f) == 0x02));
      if (is_link_local || is_multicast_link_local)
        *scope_id = if_nametoindex(if_name + 1);
      if (*scope_id == 0)
        *scope_id = atoi(if_name + 1);
    }
  }
  return result;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
}

int gethostname(char* name, int namelen, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS_RUNTIME)
  try
  {
    using namespace Windows::Foundation::Collections;
    using namespace Windows::Networking;
    using namespace Windows::Networking::Connectivity;
    IVectorView<HostName^>^ hostnames = NetworkInformation::GetHostNames();
    for (unsigned i = 0; i < hostnames->Size; ++i)
    {
      HostName^ hostname = hostnames->GetAt(i);
      if (hostname->Type == HostNameType::DomainName)
      {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string raw_name = converter.to_bytes(hostname->RawName->Data());
        if (namelen > 0 && raw_name.size() < static_cast<std::size_t>(namelen))
        {
          strcpy_s(name, namelen, raw_name.c_str());
          return 0;
        }
      }
    }
    return -1;
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
    return -1;
  }
#else // defined(ASIO_WINDOWS_RUNTIME)
  int result = error_wrapper(::gethostname(name, namelen), ec);
# if defined(ASIO_WINDOWS)
  if (result == 0)
    ec = asio::error_code();
# endif // defined(ASIO_WINDOWS)
  return result;
#endif // defined(ASIO_WINDOWS_RUNTIME)
}

#if !defined(ASIO_WINDOWS_RUNTIME)

#if !defined(ASIO_HAS_GETADDRINFO)

// The following functions are only needed for emulation of getaddrinfo and
// getnameinfo.

inline asio::error_code translate_netdb_error(int error)
{
  switch (error)
  {
  case 0:
    return asio::error_code();
  case HOST_NOT_FOUND:
    return asio::error::host_not_found;
  case TRY_AGAIN:
    return asio::error::host_not_found_try_again;
  case NO_RECOVERY:
    return asio::error::no_recovery;
  case NO_DATA:
    return asio::error::no_data;
  default:
    ASIO_ASSERT(false);
    return asio::error::invalid_argument;
  }
}

inline hostent* gethostbyaddr(const char* addr, int length, int af,
    hostent* result, char* buffer, int buflength, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  (void)(buffer);
  (void)(buflength);
  hostent* retval = error_wrapper(::gethostbyaddr(addr, length, af), ec);
  if (!retval)
    return 0;
  ec = asio::error_code();
  *result = *retval;
  return retval;
#elif defined(__sun) || defined(__QNX__)
  int error = 0;
  hostent* retval = error_wrapper(::gethostbyaddr_r(addr, length, af, result,
        buffer, buflength, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  return retval;
#elif defined(__MACH__) && defined(__APPLE__)
  (void)(buffer);
  (void)(buflength);
  int error = 0;
  hostent* retval = error_wrapper(::getipnodebyaddr(
        addr, length, af, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  if (!retval)
    return 0;
  *result = *retval;
  return retval;
#else
  hostent* retval = 0;
  int error = 0;
  error_wrapper(::gethostbyaddr_r(addr, length, af, result, buffer,
        buflength, &retval, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  return retval;
#endif
}

inline hostent* gethostbyname(const char* name, int af, struct hostent* result,
    char* buffer, int buflength, int ai_flags, asio::error_code& ec)
{
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
  (void)(buffer);
  (void)(buflength);
  (void)(ai_flags);
  if (af != ASIO_OS_DEF(AF_INET))
  {
    ec = asio::error::address_family_not_supported;
    return 0;
  }
  hostent* retval = error_wrapper(::gethostbyname(name), ec);
  if (!retval)
    return 0;
  ec = asio::error_code();
  *result = *retval;
  return result;
#elif defined(__sun) || defined(__QNX__)
  (void)(ai_flags);
  if (af != ASIO_OS_DEF(AF_INET))
  {
    ec = asio::error::address_family_not_supported;
    return 0;
  }
  int error = 0;
  hostent* retval = error_wrapper(::gethostbyname_r(name, result, buffer,
        buflength, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  return retval;
#elif defined(__MACH__) && defined(__APPLE__)
  (void)(buffer);
  (void)(buflength);
  int error = 0;
  hostent* retval = error_wrapper(::getipnodebyname(
        name, af, ai_flags, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  if (!retval)
    return 0;
  *result = *retval;
  return retval;
#else
  (void)(ai_flags);
  if (af != ASIO_OS_DEF(AF_INET))
  {
    ec = asio::error::address_family_not_supported;
    return 0;
  }
  hostent* retval = 0;
  int error = 0;
  error_wrapper(::gethostbyname_r(name, result,
        buffer, buflength, &retval, &error), ec);
  if (error)
    ec = translate_netdb_error(error);
  return retval;
#endif
}

inline void freehostent(hostent* h)
{
#if defined(__MACH__) && defined(__APPLE__)
  if (h)
    ::freehostent(h);
#else
  (void)(h);
#endif
}

// Emulation of getaddrinfo based on implementation in:
// Stevens, W. R., UNIX Network Programming Vol. 1, 2nd Ed., Prentice-Hall 1998.

struct gai_search
{
  const char* host;
  int family;
};

inline int gai_nsearch(const char* host,
    const addrinfo_type* hints, gai_search (&search)[2])
{
  int search_count = 0;
  if (host == 0 || host[0] == '\0')
  {
    if (hints->ai_flags & AI_PASSIVE)
    {
      // No host and AI_PASSIVE implies wildcard bind.
      switch (hints->ai_family)
      {
      case ASIO_OS_DEF(AF_INET):
        search[search_count].host = "0.0.0.0";
        search[search_count].family = ASIO_OS_DEF(AF_INET);
        ++search_count;
        break;
      case ASIO_OS_DEF(AF_INET6):
        search[search_count].host = "0::0";
        search[search_count].family = ASIO_OS_DEF(AF_INET6);
        ++search_count;
        break;
      case ASIO_OS_DEF(AF_UNSPEC):
        search[search_count].host = "0::0";
        search[search_count].family = ASIO_OS_DEF(AF_INET6);
        ++search_count;
        search[search_count].host = "0.0.0.0";
        search[search_count].family = ASIO_OS_DEF(AF_INET);
        ++search_count;
        break;
      default:
        break;
      }
    }
    else
    {
      // No host and not AI_PASSIVE means connect to local host.
      switch (hints->ai_family)
      {
      case ASIO_OS_DEF(AF_INET):
        search[search_count].host = "localhost";
        search[search_count].family = ASIO_OS_DEF(AF_INET);
        ++search_count;
        break;
      case ASIO_OS_DEF(AF_INET6):
        search[search_count].host = "localhost";
        search[search_count].family = ASIO_OS_DEF(AF_INET6);
        ++search_count;
        break;
      case ASIO_OS_DEF(AF_UNSPEC):
        search[search_count].host = "localhost";
        search[search_count].family = ASIO_OS_DEF(AF_INET6);
        ++search_count;
        search[search_count].host = "localhost";
        search[search_count].family = ASIO_OS_DEF(AF_INET);
        ++search_count;
        break;
      default:
        break;
      }
    }
  }
  else
  {
    // Host is specified.
    switch (hints->ai_family)
    {
    case ASIO_OS_DEF(AF_INET):
      search[search_count].host = host;
      search[search_count].family = ASIO_OS_DEF(AF_INET);
      ++search_count;
      break;
    case ASIO_OS_DEF(AF_INET6):
      search[search_count].host = host;
      search[search_count].family = ASIO_OS_DEF(AF_INET6);
      ++search_count;
      break;
    case ASIO_OS_DEF(AF_UNSPEC):
      search[search_count].host = host;
      search[search_count].family = ASIO_OS_DEF(AF_INET6);
      ++search_count;
      search[search_count].host = host;
      search[search_count].family = ASIO_OS_DEF(AF_INET);
      ++search_count;
      break;
    default:
      break;
    }
  }
  return search_count;
}

template <typename T>
inline T* gai_alloc(std::size_t size = sizeof(T))
{
  using namespace std;
  T* p = static_cast<T*>(::operator new(size, std::nothrow));
  if (p)
    memset(p, 0, size);
  return p;
}

inline void gai_free(void* p)
{
  ::operator delete(p);
}

inline void gai_strcpy(char* target, const char* source, std::size_t max_size)
{
  using namespace std;
#if defined(ASIO_HAS_SECURE_RTL)
  strcpy_s(target, max_size, source);
#else // defined(ASIO_HAS_SECURE_RTL)
  *target = 0;
  if (max_size > 0)
    strncat(target, source, max_size - 1);
#endif // defined(ASIO_HAS_SECURE_RTL)
}

enum { gai_clone_flag = 1 << 30 };

inline int gai_aistruct(addrinfo_type*** next, const addrinfo_type* hints,
    const void* addr, int family)
{
  using namespace std;

  addrinfo_type* ai = gai_alloc<addrinfo_type>();
  if (ai == 0)
    return EAI_MEMORY;

  ai->ai_next = 0;
  **next = ai;
  *next = &ai->ai_next;

  ai->ai_canonname = 0;
  ai->ai_socktype = hints->ai_socktype;
  if (ai->ai_socktype == 0)
    ai->ai_flags |= gai_clone_flag;
  ai->ai_protocol = hints->ai_protocol;
  ai->ai_family = family;

  switch (ai->ai_family)
  {
  case ASIO_OS_DEF(AF_INET):
    {
      sockaddr_in4_type* sinptr = gai_alloc<sockaddr_in4_type>();
      if (sinptr == 0)
        return EAI_MEMORY;
      sinptr->sin_family = ASIO_OS_DEF(AF_INET);
      memcpy(&sinptr->sin_addr, addr, sizeof(in4_addr_type));
      ai->ai_addr = reinterpret_cast<sockaddr*>(sinptr);
      ai->ai_addrlen = sizeof(sockaddr_in4_type);
      break;
    }
  case ASIO_OS_DEF(AF_INET6):
    {
      sockaddr_in6_type* sin6ptr = gai_alloc<sockaddr_in6_type>();
      if (sin6ptr == 0)
        return EAI_MEMORY;
      sin6ptr->sin6_family = ASIO_OS_DEF(AF_INET6);
      memcpy(&sin6ptr->sin6_addr, addr, sizeof(in6_addr_type));
      ai->ai_addr = reinterpret_cast<sockaddr*>(sin6ptr);
      ai->ai_addrlen = sizeof(sockaddr_in6_type);
      break;
    }
  default:
    break;
  }

  return 0;
}

inline addrinfo_type* gai_clone(addrinfo_type* ai)
{
  using namespace std;

  addrinfo_type* new_ai = gai_alloc<addrinfo_type>();
  if (new_ai == 0)
    return new_ai;

  new_ai->ai_next = ai->ai_next;
  ai->ai_next = new_ai;

  new_ai->ai_flags = 0;
  new_ai->ai_family = ai->ai_family;
  new_ai->ai_socktype = ai->ai_socktype;
  new_ai->ai_protocol = ai->ai_protocol;
  new_ai->ai_canonname = 0;
  new_ai->ai_addrlen = ai->ai_addrlen;
  new_ai->ai_addr = gai_alloc<sockaddr>(ai->ai_addrlen);
  memcpy(new_ai->ai_addr, ai->ai_addr, ai->ai_addrlen);

  return new_ai;
}

inline int gai_port(addrinfo_type* aihead, int port, int socktype)
{
  int num_found = 0;

  for (addrinfo_type* ai = aihead; ai; ai = ai->ai_next)
  {
    if (ai->ai_flags & gai_clone_flag)
    {
      if (ai->ai_socktype != 0)
      {
        ai = gai_clone(ai);
        if (ai == 0)
          return -1;
        // ai now points to newly cloned entry.
      }
    }
    else if (ai->ai_socktype != socktype)
    {
      // Ignore if mismatch on socket type.
      continue;
    }

    ai->ai_socktype = socktype;

    switch (ai->ai_family)
    {
    case ASIO_OS_DEF(AF_INET):
      {
        sockaddr_in4_type* sinptr =
          reinterpret_cast<sockaddr_in4_type*>(ai->ai_addr);
        sinptr->sin_port = port;
        ++num_found;
        break;
      }
    case ASIO_OS_DEF(AF_INET6):
      {
        sockaddr_in6_type* sin6ptr =
          reinterpret_cast<sockaddr_in6_type*>(ai->ai_addr);
        sin6ptr->sin6_port = port;
        ++num_found;
        break;
      }
    default:
      break;
    }
  }

  return num_found;
}

inline int gai_serv(addrinfo_type* aihead,
    const addrinfo_type* hints, const char* serv)
{
  using namespace std;

  int num_found = 0;

  if (
#if defined(AI_NUMERICSERV)
      (hints->ai_flags & AI_NUMERICSERV) ||
#endif
      isdigit(static_cast<unsigned char>(serv[0])))
  {
    int port = htons(atoi(serv));
    if (hints->ai_socktype)
    {
      // Caller specifies socket type.
      int rc = gai_port(aihead, port, hints->ai_socktype);
      if (rc < 0)
        return EAI_MEMORY;
      num_found += rc;
    }
    else
    {
      // Caller does not specify socket type.
      int rc = gai_port(aihead, port, SOCK_STREAM);
      if (rc < 0)
        return EAI_MEMORY;
      num_found += rc;
      rc = gai_port(aihead, port, SOCK_DGRAM);
      if (rc < 0)
        return EAI_MEMORY;
      num_found += rc;
    }
  }
  else
  {
    // Try service name with TCP first, then UDP.
    if (hints->ai_socktype == 0 || hints->ai_socktype == SOCK_STREAM)
    {
      servent* sptr = getservbyname(serv, "tcp");
      if (sptr != 0)
      {
        int rc = gai_port(aihead, sptr->s_port, SOCK_STREAM);
        if (rc < 0)
          return EAI_MEMORY;
        num_found += rc;
      }
    }
    if (hints->ai_socktype == 0 || hints->ai_socktype == SOCK_DGRAM)
    {
      servent* sptr = getservbyname(serv, "udp");
      if (sptr != 0)
      {
        int rc = gai_port(aihead, sptr->s_port, SOCK_DGRAM);
        if (rc < 0)
          return EAI_MEMORY;
        num_found += rc;
      }
    }
  }

  if (num_found == 0)
  {
    if (hints->ai_socktype == 0)
    {
      // All calls to getservbyname() failed.
      return EAI_NONAME;
    }
    else
    {
      // Service not supported for socket type.
      return EAI_SERVICE;
    }
  }

  return 0;
}

inline int gai_echeck(const char* host, const char* service,
    int flags, int family, int socktype, int protocol)
{
  (void)(flags);
  (void)(protocol);

  // Host or service must be specified.
  if (host == 0 || host[0] == '\0')
    if (service == 0 || service[0] == '\0')
      return EAI_NONAME;

  // Check combination of family and socket type.
  switch (family)
  {
  case ASIO_OS_DEF(AF_UNSPEC):
    break;
  case ASIO_OS_DEF(AF_INET):
  case ASIO_OS_DEF(AF_INET6):
    if (service != 0 && service[0] != '\0')
      if (socktype != 0 && socktype != SOCK_STREAM && socktype != SOCK_DGRAM)
        return EAI_SOCKTYPE;
    break;
  default:
    return EAI_FAMILY;
  }

  return 0;
}

inline void freeaddrinfo_emulation(addrinfo_type* aihead)
{
  addrinfo_type* ai = aihead;
  while (ai)
  {
    gai_free(ai->ai_addr);
    gai_free(ai->ai_canonname);
    addrinfo_type* ainext = ai->ai_next;
    gai_free(ai);
    ai = ainext;
  }
}

inline int getaddrinfo_emulation(const char* host, const char* service,
    const addrinfo_type* hintsp, addrinfo_type** result)
{
  // Set up linked list of addrinfo structures.
  addrinfo_type* aihead = 0;
  addrinfo_type** ainext = &aihead;
  char* canon = 0;

  // Supply default hints if not specified by caller.
  addrinfo_type hints = addrinfo_type();
  hints.ai_family = ASIO_OS_DEF(AF_UNSPEC);
  if (hintsp)
    hints = *hintsp;

  // If the resolution is not specifically for AF_INET6, remove the AI_V4MAPPED
  // and AI_ALL flags.
#if defined(AI_V4MAPPED)
  if (hints.ai_family != ASIO_OS_DEF(AF_INET6))
    hints.ai_flags &= ~AI_V4MAPPED;
#endif
#if defined(AI_ALL)
  if (hints.ai_family != ASIO_OS_DEF(AF_INET6))
    hints.ai_flags &= ~AI_ALL;
#endif

  // Basic error checking.
  int rc = gai_echeck(host, service, hints.ai_flags, hints.ai_family,
      hints.ai_socktype, hints.ai_protocol);
  if (rc != 0)
  {
    freeaddrinfo_emulation(aihead);
    return rc;
  }

  gai_search search[2];
  int search_count = gai_nsearch(host, &hints, search);
  for (gai_search* sptr = search; sptr < search + search_count; ++sptr)
  {
    // Check for IPv4 dotted decimal string.
    in4_addr_type inaddr;
    asio::error_code ec;
    if (socket_ops::inet_pton(ASIO_OS_DEF(AF_INET),
          sptr->host, &inaddr, 0, ec) == 1)
    {
      if (hints.ai_family != ASIO_OS_DEF(AF_UNSPEC)
          && hints.ai_family != ASIO_OS_DEF(AF_INET))
      {
        freeaddrinfo_emulation(aihead);
        gai_free(canon);
        return EAI_FAMILY;
      }
      if (sptr->family == ASIO_OS_DEF(AF_INET))
      {
        rc = gai_aistruct(&ainext, &hints, &inaddr, ASIO_OS_DEF(AF_INET));
        if (rc != 0)
        {
          freeaddrinfo_emulation(aihead);
          gai_free(canon);
          return rc;
        }
      }
      continue;
    }

    // Check for IPv6 hex string.
    in6_addr_type in6addr;
    if (socket_ops::inet_pton(ASIO_OS_DEF(AF_INET6),
          sptr->host, &in6addr, 0, ec) == 1)
    {
      if (hints.ai_family != ASIO_OS_DEF(AF_UNSPEC)
          && hints.ai_family != ASIO_OS_DEF(AF_INET6))
      {
        freeaddrinfo_emulation(aihead);
        gai_free(canon);
        return EAI_FAMILY;
      }
      if (sptr->family == ASIO_OS_DEF(AF_INET6))
      {
        rc = gai_aistruct(&ainext, &hints, &in6addr,
            ASIO_OS_DEF(AF_INET6));
        if (rc != 0)
        {
          freeaddrinfo_emulation(aihead);
          gai_free(canon);
          return rc;
        }
      }
      continue;
    }

    // Look up hostname.
    hostent hent;
    char hbuf[8192] = "";
    hostent* hptr = socket_ops::gethostbyname(sptr->host,
        sptr->family, &hent, hbuf, sizeof(hbuf), hints.ai_flags, ec);
    if (hptr == 0)
    {
      if (search_count == 2)
      {
        // Failure is OK if there are multiple searches.
        continue;
      }
      freeaddrinfo_emulation(aihead);
      gai_free(canon);
      if (ec == asio::error::host_not_found)
        return EAI_NONAME;
      if (ec == asio::error::host_not_found_try_again)
        return EAI_AGAIN;
      if (ec == asio::error::no_recovery)
        return EAI_FAIL;
      if (ec == asio::error::no_data)
        return EAI_NONAME;
      return EAI_NONAME;
    }

    // Check for address family mismatch if one was specified.
    if (hints.ai_family != ASIO_OS_DEF(AF_UNSPEC)
        && hints.ai_family != hptr->h_addrtype)
    {
      freeaddrinfo_emulation(aihead);
      gai_free(canon);
      socket_ops::freehostent(hptr);
      return EAI_FAMILY;
    }

    // Save canonical name first time.
    if (host != 0 && host[0] != '\0' && hptr->h_name && hptr->h_name[0]
        && (hints.ai_flags & AI_CANONNAME) && canon == 0)
    {
      std::size_t canon_len = strlen(hptr->h_name) + 1;
      canon = gai_alloc<char>(canon_len);
      if (canon == 0)
      {
        freeaddrinfo_emulation(aihead);
        socket_ops::freehostent(hptr);
        return EAI_MEMORY;
      }
      gai_strcpy(canon, hptr->h_name, canon_len);
    }

    // Create an addrinfo structure for each returned address.
    for (char** ap = hptr->h_addr_list; *ap; ++ap)
    {
      rc = gai_aistruct(&ainext, &hints, *ap, hptr->h_addrtype);
      if (rc != 0)
      {
        freeaddrinfo_emulation(aihead);
        gai_free(canon);
        socket_ops::freehostent(hptr);
        return EAI_FAMILY;
      }
    }

    socket_ops::freehostent(hptr);
  }

  // Check if we found anything.
  if (aihead == 0)
  {
    gai_free(canon);
    return EAI_NONAME;
  }

  // Return canonical name in first entry.
  if (host != 0 && host[0] != '\0' && (hints.ai_flags & AI_CANONNAME))
  {
    if (canon)
    {
      aihead->ai_canonname = canon;
      canon = 0;
    }
    else
    {
      std::size_t canonname_len = strlen(search[0].host) + 1;
      aihead->ai_canonname = gai_alloc<char>(canonname_len);
      if (aihead->ai_canonname == 0)
      {
        freeaddrinfo_emulation(aihead);
        return EAI_MEMORY;
      }
      gai_strcpy(aihead->ai_canonname, search[0].host, canonname_len);
    }
  }
  gai_free(canon);

  // Process the service name.
  if (service != 0 && service[0] != '\0')
  {
    rc = gai_serv(aihead, &hints, service);
    if (rc != 0)
    {
      freeaddrinfo_emulation(aihead);
      return rc;
    }
  }

  // Return result to caller.
  *result = aihead;
  return 0;
}

inline asio::error_code getnameinfo_emulation(
    const socket_addr_type* sa, std::size_t salen, char* host,
    std::size_t hostlen, char* serv, std::size_t servlen, int flags,
    asio::error_code& ec)
{
  using namespace std;

  const char* addr;
  size_t addr_len;
  unsigned short port;
  switch (sa->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    if (salen != sizeof(sockaddr_in4_type))
    {
      return ec = asio::error::invalid_argument;
    }
    addr = reinterpret_cast<const char*>(
        &reinterpret_cast<const sockaddr_in4_type*>(sa)->sin_addr);
    addr_len = sizeof(in4_addr_type);
    port = reinterpret_cast<const sockaddr_in4_type*>(sa)->sin_port;
    break;
  case ASIO_OS_DEF(AF_INET6):
    if (salen != sizeof(sockaddr_in6_type))
    {
      return ec = asio::error::invalid_argument;
    }
    addr = reinterpret_cast<const char*>(
        &reinterpret_cast<const sockaddr_in6_type*>(sa)->sin6_addr);
    addr_len = sizeof(in6_addr_type);
    port = reinterpret_cast<const sockaddr_in6_type*>(sa)->sin6_port;
    break;
  default:
    return ec = asio::error::address_family_not_supported;
  }

  if (host && hostlen > 0)
  {
    if (flags & NI_NUMERICHOST)
    {
      if (socket_ops::inet_ntop(sa->sa_family, addr, host, hostlen, 0, ec) == 0)
      {
        return ec;
      }
    }
    else
    {
      hostent hent;
      char hbuf[8192] = "";
      hostent* hptr = socket_ops::gethostbyaddr(addr,
          static_cast<int>(addr_len), sa->sa_family,
          &hent, hbuf, sizeof(hbuf), ec);
      if (hptr && hptr->h_name && hptr->h_name[0] != '\0')
      {
        if (flags & NI_NOFQDN)
        {
          char* dot = strchr(hptr->h_name, '.');
          if (dot)
          {
            *dot = 0;
          }
        }
        gai_strcpy(host, hptr->h_name, hostlen);
        socket_ops::freehostent(hptr);
      }
      else
      {
        socket_ops::freehostent(hptr);
        if (flags & NI_NAMEREQD)
        {
          return ec = asio::error::host_not_found;
        }
        if (socket_ops::inet_ntop(sa->sa_family,
              addr, host, hostlen, 0, ec) == 0)
        {
          return ec;
        }
      }
    }
  }

  if (serv && servlen > 0)
  {
    if (flags & NI_NUMERICSERV)
    {
      if (servlen < 6)
      {
        return ec = asio::error::no_buffer_space;
      }
#if defined(ASIO_HAS_SECURE_RTL)
      sprintf_s(serv, servlen, "%u", ntohs(port));
#else // defined(ASIO_HAS_SECURE_RTL)
      sprintf(serv, "%u", ntohs(port));
#endif // defined(ASIO_HAS_SECURE_RTL)
    }
    else
    {
#if defined(ASIO_HAS_PTHREADS)
      static ::pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
      ::pthread_mutex_lock(&mutex);
#endif // defined(ASIO_HAS_PTHREADS)
      servent* sptr = ::getservbyport(port, (flags & NI_DGRAM) ? "udp" : 0);
      if (sptr && sptr->s_name && sptr->s_name[0] != '\0')
      {
        gai_strcpy(serv, sptr->s_name, servlen);
      }
      else
      {
        if (servlen < 6)
        {
          return ec = asio::error::no_buffer_space;
        }
#if defined(ASIO_HAS_SECURE_RTL)
        sprintf_s(serv, servlen, "%u", ntohs(port));
#else // defined(ASIO_HAS_SECURE_RTL)
        sprintf(serv, "%u", ntohs(port));
#endif // defined(ASIO_HAS_SECURE_RTL)
      }
#if defined(ASIO_HAS_PTHREADS)
      ::pthread_mutex_unlock(&mutex);
#endif // defined(ASIO_HAS_PTHREADS)
    }
  }

  ec = asio::error_code();
  return ec;
}

#endif // !defined(ASIO_HAS_GETADDRINFO)

inline asio::error_code translate_addrinfo_error(int error)
{
  switch (error)
  {
  case 0:
    return asio::error_code();
  case EAI_AGAIN:
    return asio::error::host_not_found_try_again;
  case EAI_BADFLAGS:
    return asio::error::invalid_argument;
  case EAI_FAIL:
    return asio::error::no_recovery;
  case EAI_FAMILY:
    return asio::error::address_family_not_supported;
  case EAI_MEMORY:
    return asio::error::no_memory;
  case EAI_NONAME:
#if defined(EAI_ADDRFAMILY)
  case EAI_ADDRFAMILY:
#endif
#if defined(EAI_NODATA) && (EAI_NODATA != EAI_NONAME)
  case EAI_NODATA:
#endif
    return asio::error::host_not_found;
  case EAI_SERVICE:
    return asio::error::service_not_found;
  case EAI_SOCKTYPE:
    return asio::error::socket_type_not_supported;
  default: // Possibly the non-portable EAI_SYSTEM.
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
    return asio::error_code(
        WSAGetLastError(), asio::error::get_system_category());
#else
    return asio::error_code(
        errno, asio::error::get_system_category());
#endif
  }
}

asio::error_code getaddrinfo(const char* host,
    const char* service, const addrinfo_type& hints,
    addrinfo_type** result, asio::error_code& ec)
{
  host = (host && *host) ? host : 0;
  service = (service && *service) ? service : 0;
  clear_last_error();
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# if defined(ASIO_HAS_GETADDRINFO)
  // Building for Windows XP, Windows Server 2003, or later.
  int error = ::getaddrinfo(host, service, &hints, result);
  return ec = translate_addrinfo_error(error);
# else
  // Building for Windows 2000 or earlier.
  typedef int (WSAAPI *gai_t)(const char*,
      const char*, const addrinfo_type*, addrinfo_type**);
  if (HMODULE winsock_module = ::GetModuleHandleA("ws2_32"))
  {
    if (gai_t gai = (gai_t)::GetProcAddress(winsock_module, "getaddrinfo"))
    {
      int error = gai(host, service, &hints, result);
      return ec = translate_addrinfo_error(error);
    }
  }
  int error = getaddrinfo_emulation(host, service, &hints, result);
  return ec = translate_addrinfo_error(error);
# endif
#elif !defined(ASIO_HAS_GETADDRINFO)
  int error = getaddrinfo_emulation(host, service, &hints, result);
  return ec = translate_addrinfo_error(error);
#else
  int error = ::getaddrinfo(host, service, &hints, result);
#if defined(__MACH__) && defined(__APPLE__)
  using namespace std; // For isdigit and atoi.
  if (error == 0 && service && isdigit(static_cast<unsigned char>(service[0])))
  {
    u_short_type port = host_to_network_short(atoi(service));
    for (addrinfo_type* ai = *result; ai; ai = ai->ai_next)
    {
      switch (ai->ai_family)
      {
      case ASIO_OS_DEF(AF_INET):
        {
          sockaddr_in4_type* sinptr =
            reinterpret_cast<sockaddr_in4_type*>(ai->ai_addr);
          if (sinptr->sin_port == 0)
            sinptr->sin_port = port;
          break;
        }
      case ASIO_OS_DEF(AF_INET6):
        {
          sockaddr_in6_type* sin6ptr =
            reinterpret_cast<sockaddr_in6_type*>(ai->ai_addr);
          if (sin6ptr->sin6_port == 0)
            sin6ptr->sin6_port = port;
          break;
        }
      default:
        break;
      }
    }
  }
#endif
  return ec = translate_addrinfo_error(error);
#endif
}

asio::error_code background_getaddrinfo(
    const weak_cancel_token_type& cancel_token, const char* host,
    const char* service, const addrinfo_type& hints,
    addrinfo_type** result, asio::error_code& ec)
{
  if (cancel_token.expired())
    ec = asio::error::operation_aborted;
  else
    socket_ops::getaddrinfo(host, service, hints, result, ec);
  return ec;
}

void freeaddrinfo(addrinfo_type* ai)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# if defined(ASIO_HAS_GETADDRINFO)
  // Building for Windows XP, Windows Server 2003, or later.
  ::freeaddrinfo(ai);
# else
  // Building for Windows 2000 or earlier.
  typedef int (WSAAPI *fai_t)(addrinfo_type*);
  if (HMODULE winsock_module = ::GetModuleHandleA("ws2_32"))
  {
    if (fai_t fai = (fai_t)::GetProcAddress(winsock_module, "freeaddrinfo"))
    {
      fai(ai);
      return;
    }
  }
  freeaddrinfo_emulation(ai);
# endif
#elif !defined(ASIO_HAS_GETADDRINFO)
  freeaddrinfo_emulation(ai);
#else
  ::freeaddrinfo(ai);
#endif
}

asio::error_code getnameinfo(const socket_addr_type* addr,
    std::size_t addrlen, char* host, std::size_t hostlen,
    char* serv, std::size_t servlen, int flags, asio::error_code& ec)
{
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# if defined(ASIO_HAS_GETADDRINFO)
  // Building for Windows XP, Windows Server 2003, or later.
  clear_last_error();
  int error = ::getnameinfo(addr, static_cast<socklen_t>(addrlen),
      host, static_cast<DWORD>(hostlen),
      serv, static_cast<DWORD>(servlen), flags);
  return ec = translate_addrinfo_error(error);
# else
  // Building for Windows 2000 or earlier.
  typedef int (WSAAPI *gni_t)(const socket_addr_type*,
      int, char*, DWORD, char*, DWORD, int);
  if (HMODULE winsock_module = ::GetModuleHandleA("ws2_32"))
  {
    if (gni_t gni = (gni_t)::GetProcAddress(winsock_module, "getnameinfo"))
    {
      clear_last_error();
      int error = gni(addr, static_cast<int>(addrlen),
          host, static_cast<DWORD>(hostlen),
          serv, static_cast<DWORD>(servlen), flags);
      return ec = translate_addrinfo_error(error);
    }
  }
  clear_last_error();
  return getnameinfo_emulation(addr, addrlen,
      host, hostlen, serv, servlen, flags, ec);
# endif
#elif !defined(ASIO_HAS_GETADDRINFO)
  using namespace std; // For memcpy.
  sockaddr_storage_type tmp_addr;
  memcpy(&tmp_addr, addr, addrlen);
  addr = reinterpret_cast<socket_addr_type*>(&tmp_addr);
  clear_last_error();
  return getnameinfo_emulation(addr, addrlen,
      host, hostlen, serv, servlen, flags, ec);
#else
  clear_last_error();
  int error = ::getnameinfo(addr, addrlen, host, hostlen, serv, servlen, flags);
  return ec = translate_addrinfo_error(error);
#endif
}

asio::error_code sync_getnameinfo(
    const socket_addr_type* addr, std::size_t addrlen,
    char* host, std::size_t hostlen, char* serv,
    std::size_t servlen, int sock_type, asio::error_code& ec)
{
  // First try resolving with the service name. If that fails try resolving
  // but allow the service to be returned as a number.
  int flags = (sock_type == SOCK_DGRAM) ? NI_DGRAM : 0;
  socket_ops::getnameinfo(addr, addrlen, host,
      hostlen, serv, servlen, flags, ec);
  if (ec)
  {
    socket_ops::getnameinfo(addr, addrlen, host, hostlen,
        serv, servlen, flags | NI_NUMERICSERV, ec);
  }

  return ec;
}

asio::error_code background_getnameinfo(
    const weak_cancel_token_type& cancel_token,
    const socket_addr_type* addr, std::size_t addrlen,
    char* host, std::size_t hostlen, char* serv,
    std::size_t servlen, int sock_type, asio::error_code& ec)
{
  if (cancel_token.expired())
  {
    ec = asio::error::operation_aborted;
  }
  else
  {
    // First try resolving with the service name. If that fails try resolving
    // but allow the service to be returned as a number.
    int flags = (sock_type == SOCK_DGRAM) ? NI_DGRAM : 0;
    socket_ops::getnameinfo(addr, addrlen, host,
        hostlen, serv, servlen, flags, ec);
    if (ec)
    {
      socket_ops::getnameinfo(addr, addrlen, host, hostlen,
          serv, servlen, flags | NI_NUMERICSERV, ec);
    }
  }

  return ec;
}

#endif // !defined(ASIO_WINDOWS_RUNTIME)

u_long_type network_to_host_long(u_long_type value)
{
#if defined(ASIO_WINDOWS_RUNTIME)
  unsigned char* value_p = reinterpret_cast<unsigned char*>(&value);
  u_long_type result = (static_cast<u_long_type>(value_p[0]) << 24)
    | (static_cast<u_long_type>(value_p[1]) << 16)
    | (static_cast<u_long_type>(value_p[2]) << 8)
    | static_cast<u_long_type>(value_p[3]);
  return result;
#else // defined(ASIO_WINDOWS_RUNTIME)
  return ntohl(value);
#endif // defined(ASIO_WINDOWS_RUNTIME)
}

u_long_type host_to_network_long(u_long_type value)
{
#if defined(ASIO_WINDOWS_RUNTIME)
  u_long_type result;
  unsigned char* result_p = reinterpret_cast<unsigned char*>(&result);
  result_p[0] = static_cast<unsigned char>((value >> 24) & 0xFF);
  result_p[1] = static_cast<unsigned char>((value >> 16) & 0xFF);
  result_p[2] = static_cast<unsigned char>((value >> 8) & 0xFF);
  result_p[3] = static_cast<unsigned char>(value & 0xFF);
  return result;
#else // defined(ASIO_WINDOWS_RUNTIME)
  return htonl(value);
#endif // defined(ASIO_WINDOWS_RUNTIME)
}

u_short_type network_to_host_short(u_short_type value)
{
#if defined(ASIO_WINDOWS_RUNTIME)
  unsigned char* value_p = reinterpret_cast<unsigned char*>(&value);
  u_short_type result = (static_cast<u_short_type>(value_p[0]) << 8)
    | static_cast<u_short_type>(value_p[1]);
  return result;
#else // defined(ASIO_WINDOWS_RUNTIME)
  return ntohs(value);
#endif // defined(ASIO_WINDOWS_RUNTIME)
}

u_short_type host_to_network_short(u_short_type value)
{
#if defined(ASIO_WINDOWS_RUNTIME)
  u_short_type result;
  unsigned char* result_p = reinterpret_cast<unsigned char*>(&result);
  result_p[0] = static_cast<unsigned char>((value >> 8) & 0xFF);
  result_p[1] = static_cast<unsigned char>(value & 0xFF);
  return result;
#else // defined(ASIO_WINDOWS_RUNTIME)
  return htons(value);
#endif // defined(ASIO_WINDOWS_RUNTIME)
}

} // namespace socket_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_OPS_IPP
