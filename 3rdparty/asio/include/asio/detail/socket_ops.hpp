//
// detail/socket_ops.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_OPS_HPP
#define ASIO_DETAIL_SOCKET_OPS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/error_code.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace socket_ops {

// Socket state bits.
enum
{
  // The user wants a non-blocking socket.
  user_set_non_blocking = 1,

  // The socket has been set non-blocking.
  internal_non_blocking = 2,

  // Helper "state" used to determine whether the socket is non-blocking.
  non_blocking = user_set_non_blocking | internal_non_blocking,

  // User wants connection_aborted errors, which are disabled by default.
  enable_connection_aborted = 4,

  // The user set the linger option. Needs to be checked when closing.
  user_set_linger = 8,

  // The socket is stream-oriented.
  stream_oriented = 16,

  // The socket is datagram-oriented.
  datagram_oriented = 32,

  // The socket may have been dup()-ed.
  possible_dup = 64
};

typedef unsigned char state_type;

struct noop_deleter { void operator()(void*) {} };
typedef shared_ptr<void> shared_cancel_token_type;
typedef weak_ptr<void> weak_cancel_token_type;

#if !defined(ASIO_WINDOWS_RUNTIME)

ASIO_DECL socket_type accept(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec);

ASIO_DECL socket_type sync_accept(socket_type s,
    state_type state, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_accept(socket_type s,
    void* output_buffer, DWORD address_length,
    socket_addr_type* addr, std::size_t* addrlen,
    socket_type new_socket, asio::error_code& ec);

#else // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_accept(socket_type s,
    state_type state, socket_addr_type* addr, std::size_t* addrlen,
    asio::error_code& ec, socket_type& new_socket);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL int bind(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec);

ASIO_DECL int close(socket_type s, state_type& state,
    bool destruction, asio::error_code& ec);

ASIO_DECL bool set_user_non_blocking(socket_type s,
    state_type& state, bool value, asio::error_code& ec);

ASIO_DECL bool set_internal_non_blocking(socket_type s,
    state_type& state, bool value, asio::error_code& ec);

ASIO_DECL int shutdown(socket_type s,
    int what, asio::error_code& ec);

ASIO_DECL int connect(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec);

ASIO_DECL void sync_connect(socket_type s, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_connect(socket_type s,
    asio::error_code& ec);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_connect(socket_type s,
    asio::error_code& ec);

ASIO_DECL int socketpair(int af, int type, int protocol,
    socket_type sv[2], asio::error_code& ec);

ASIO_DECL bool sockatmark(socket_type s, asio::error_code& ec);

ASIO_DECL size_t available(socket_type s, asio::error_code& ec);

ASIO_DECL int listen(socket_type s,
    int backlog, asio::error_code& ec);

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef WSABUF buf;
#else // defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef iovec buf;
#endif // defined(ASIO_WINDOWS) || defined(__CYGWIN__)

ASIO_DECL void init_buf(buf& b, void* data, size_t size);

ASIO_DECL void init_buf(buf& b, const void* data, size_t size);

ASIO_DECL signed_size_type recv(socket_type s, buf* bufs,
    size_t count, int flags, asio::error_code& ec);

ASIO_DECL size_t sync_recv(socket_type s, state_type state, buf* bufs,
    size_t count, int flags, bool all_empty, asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_recv(state_type state,
    const weak_cancel_token_type& cancel_token, bool all_empty,
    asio::error_code& ec, size_t bytes_transferred);

#else // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_recv(socket_type s,
    buf* bufs, size_t count, int flags, bool is_stream,
    asio::error_code& ec, size_t& bytes_transferred);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL signed_size_type recvfrom(socket_type s, buf* bufs,
    size_t count, int flags, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec);

ASIO_DECL size_t sync_recvfrom(socket_type s, state_type state,
    buf* bufs, size_t count, int flags, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_recvfrom(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec);

#else // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_recvfrom(socket_type s,
    buf* bufs, size_t count, int flags,
    socket_addr_type* addr, std::size_t* addrlen,
    asio::error_code& ec, size_t& bytes_transferred);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL signed_size_type recvmsg(socket_type s, buf* bufs,
    size_t count, int in_flags, int& out_flags,
    asio::error_code& ec);

ASIO_DECL size_t sync_recvmsg(socket_type s, state_type state,
    buf* bufs, size_t count, int in_flags, int& out_flags,
    asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_recvmsg(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec);

#else // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_recvmsg(socket_type s,
    buf* bufs, size_t count, int in_flags, int& out_flags,
    asio::error_code& ec, size_t& bytes_transferred);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL signed_size_type send(socket_type s, const buf* bufs,
    size_t count, int flags, asio::error_code& ec);

ASIO_DECL size_t sync_send(socket_type s, state_type state,
    const buf* bufs, size_t count, int flags,
    bool all_empty, asio::error_code& ec);

#if defined(ASIO_HAS_IOCP)

ASIO_DECL void complete_iocp_send(
    const weak_cancel_token_type& cancel_token,
    asio::error_code& ec);

#else // defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_send(socket_type s,
    const buf* bufs, size_t count, int flags,
    asio::error_code& ec, size_t& bytes_transferred);

#endif // defined(ASIO_HAS_IOCP)

ASIO_DECL signed_size_type sendto(socket_type s, const buf* bufs,
    size_t count, int flags, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec);

ASIO_DECL size_t sync_sendto(socket_type s, state_type state,
    const buf* bufs, size_t count, int flags, const socket_addr_type* addr,
    std::size_t addrlen, asio::error_code& ec);

#if !defined(ASIO_HAS_IOCP)

ASIO_DECL bool non_blocking_sendto(socket_type s,
    const buf* bufs, size_t count, int flags,
    const socket_addr_type* addr, std::size_t addrlen,
    asio::error_code& ec, size_t& bytes_transferred);

#endif // !defined(ASIO_HAS_IOCP)

ASIO_DECL socket_type socket(int af, int type, int protocol,
    asio::error_code& ec);

ASIO_DECL int setsockopt(socket_type s, state_type& state,
    int level, int optname, const void* optval,
    std::size_t optlen, asio::error_code& ec);

ASIO_DECL int getsockopt(socket_type s, state_type state,
    int level, int optname, void* optval,
    size_t* optlen, asio::error_code& ec);

ASIO_DECL int getpeername(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, bool cached, asio::error_code& ec);

ASIO_DECL int getsockname(socket_type s, socket_addr_type* addr,
    std::size_t* addrlen, asio::error_code& ec);

ASIO_DECL int ioctl(socket_type s, state_type& state,
    int cmd, ioctl_arg_type* arg, asio::error_code& ec);

ASIO_DECL int select(int nfds, fd_set* readfds, fd_set* writefds,
    fd_set* exceptfds, timeval* timeout, asio::error_code& ec);

ASIO_DECL int poll_read(socket_type s,
    state_type state, int msec, asio::error_code& ec);

ASIO_DECL int poll_write(socket_type s,
    state_type state, int msec, asio::error_code& ec);

ASIO_DECL int poll_error(socket_type s,
    state_type state, int msec, asio::error_code& ec);

ASIO_DECL int poll_connect(socket_type s,
    int msec, asio::error_code& ec);

#endif // !defined(ASIO_WINDOWS_RUNTIME)

ASIO_DECL const char* inet_ntop(int af, const void* src, char* dest,
    size_t length, unsigned long scope_id, asio::error_code& ec);

ASIO_DECL int inet_pton(int af, const char* src, void* dest,
    unsigned long* scope_id, asio::error_code& ec);

ASIO_DECL int gethostname(char* name,
    int namelen, asio::error_code& ec);

#if !defined(ASIO_WINDOWS_RUNTIME)

ASIO_DECL asio::error_code getaddrinfo(const char* host,
    const char* service, const addrinfo_type& hints,
    addrinfo_type** result, asio::error_code& ec);

ASIO_DECL asio::error_code background_getaddrinfo(
    const weak_cancel_token_type& cancel_token, const char* host,
    const char* service, const addrinfo_type& hints,
    addrinfo_type** result, asio::error_code& ec);

ASIO_DECL void freeaddrinfo(addrinfo_type* ai);

ASIO_DECL asio::error_code getnameinfo(
    const socket_addr_type* addr, std::size_t addrlen,
    char* host, std::size_t hostlen, char* serv,
    std::size_t servlen, int flags, asio::error_code& ec);

ASIO_DECL asio::error_code sync_getnameinfo(
    const socket_addr_type* addr, std::size_t addrlen,
    char* host, std::size_t hostlen, char* serv,
    std::size_t servlen, int sock_type, asio::error_code& ec);

ASIO_DECL asio::error_code background_getnameinfo(
    const weak_cancel_token_type& cancel_token,
    const socket_addr_type* addr, std::size_t addrlen,
    char* host, std::size_t hostlen, char* serv,
    std::size_t servlen, int sock_type, asio::error_code& ec);

#endif // !defined(ASIO_WINDOWS_RUNTIME)

ASIO_DECL u_long_type network_to_host_long(u_long_type value);

ASIO_DECL u_long_type host_to_network_long(u_long_type value);

ASIO_DECL u_short_type network_to_host_short(u_short_type value);

ASIO_DECL u_short_type host_to_network_short(u_short_type value);

} // namespace socket_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/socket_ops.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_SOCKET_OPS_HPP
