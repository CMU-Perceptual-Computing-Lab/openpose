//
// detail/socket_types.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_TYPES_HPP
#define ASIO_DETAIL_SOCKET_TYPES_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
// Empty.
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# if defined(_WINSOCKAPI_) && !defined(_WINSOCK2API_)
#  error WinSock.h has already been included
# endif // defined(_WINSOCKAPI_) && !defined(_WINSOCK2API_)
# if defined(__BORLANDC__)
#  include <stdlib.h> // Needed for __errno
#  if !defined(_WSPIAPI_H_)
#   define _WSPIAPI_H_
#   define ASIO_WSPIAPI_H_DEFINED
#  endif // !defined(_WSPIAPI_H_)
# endif // defined(__BORLANDC__)
# include <winsock2.h>
# include <ws2tcpip.h>
# if defined(WINAPI_FAMILY)
#  if ((WINAPI_FAMILY & WINAPI_PARTITION_DESKTOP) != 0)
#   include <windows.h>
#  endif // ((WINAPI_FAMILY & WINAPI_PARTITION_DESKTOP) != 0)
# endif // defined(WINAPI_FAMILY)
# if !defined(ASIO_WINDOWS_APP)
#  include <mswsock.h>
# endif // !defined(ASIO_WINDOWS_APP)
# if defined(ASIO_WSPIAPI_H_DEFINED)
#  undef _WSPIAPI_H_
#  undef ASIO_WSPIAPI_H_DEFINED
# endif // defined(ASIO_WSPIAPI_H_DEFINED)
# if !defined(ASIO_NO_DEFAULT_LINKED_LIBS)
#  if defined(UNDER_CE)
#   pragma comment(lib, "ws2.lib")
#  elif defined(_MSC_VER) || defined(__BORLANDC__)
#   pragma comment(lib, "ws2_32.lib")
#   if !defined(ASIO_WINDOWS_APP)
#    pragma comment(lib, "mswsock.lib")
#   endif // !defined(ASIO_WINDOWS_APP)
#  endif // defined(_MSC_VER) || defined(__BORLANDC__)
# endif // !defined(ASIO_NO_DEFAULT_LINKED_LIBS)
# include "asio/detail/old_win_sdk_compat.hpp"
#else
# include <sys/ioctl.h>
# if (defined(__MACH__) && defined(__APPLE__)) \
   || defined(__FreeBSD__) || defined(__NetBSD__) \
   || defined(__OpenBSD__) || defined(__linux__) \
   || defined(__EMSCRIPTEN__)
#  include <poll.h>
# elif !defined(__SYMBIAN32__)
#  include <sys/poll.h>
# endif
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
# if defined(__hpux)
#  include <sys/time.h>
# endif
# if !defined(__hpux) || defined(__SELECT)
#  include <sys/select.h>
# endif
# include <sys/socket.h>
# include <sys/uio.h>
# include <sys/un.h>
# include <netinet/in.h>
# if !defined(__SYMBIAN32__)
#  include <netinet/tcp.h>
# endif
# include <arpa/inet.h>
# include <netdb.h>
# include <net/if.h>
# include <limits.h>
# if defined(__sun)
#  include <sys/filio.h>
#  include <sys/sockio.h>
# endif
#endif

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS_RUNTIME)
const int max_addr_v4_str_len = 256;
const int max_addr_v6_str_len = 256;
typedef unsigned __int32 u_long_type;
typedef unsigned __int16 u_short_type;
struct in4_addr_type { u_long_type s_addr; };
struct in4_mreq_type { in4_addr_type imr_multiaddr, imr_interface; };
struct in6_addr_type { unsigned char s6_addr[16]; };
struct in6_mreq_type { in6_addr_type ipv6mr_multiaddr;
  unsigned long ipv6mr_interface; };
struct socket_addr_type { int sa_family; };
struct sockaddr_in4_type { int sin_family;
  in4_addr_type sin_addr; u_short_type sin_port; };
struct sockaddr_in6_type { int sin6_family;
  in6_addr_type sin6_addr; u_short_type sin6_port;
  u_long_type sin6_flowinfo; u_long_type sin6_scope_id; };
struct sockaddr_storage_type { int ss_family;
  unsigned char ss_bytes[128 - sizeof(int)]; };
struct addrinfo_type { int ai_flags;
  int ai_family, ai_socktype, ai_protocol;
  int ai_addrlen; const void* ai_addr;
  const char* ai_canonname; addrinfo_type* ai_next; };
struct linger_type { u_short_type l_onoff, l_linger; };
typedef u_long_type ioctl_arg_type;
typedef int signed_size_type;
# define ASIO_OS_DEF(c) ASIO_OS_DEF_##c
# define ASIO_OS_DEF_AF_UNSPEC 0
# define ASIO_OS_DEF_AF_INET 2
# define ASIO_OS_DEF_AF_INET6 23
# define ASIO_OS_DEF_SOCK_STREAM 1
# define ASIO_OS_DEF_SOCK_DGRAM 2
# define ASIO_OS_DEF_SOCK_RAW 3
# define ASIO_OS_DEF_SOCK_SEQPACKET 5
# define ASIO_OS_DEF_IPPROTO_IP 0
# define ASIO_OS_DEF_IPPROTO_IPV6 41
# define ASIO_OS_DEF_IPPROTO_TCP 6
# define ASIO_OS_DEF_IPPROTO_UDP 17
# define ASIO_OS_DEF_IPPROTO_ICMP 1
# define ASIO_OS_DEF_IPPROTO_ICMPV6 58
# define ASIO_OS_DEF_FIONBIO 1
# define ASIO_OS_DEF_FIONREAD 2
# define ASIO_OS_DEF_INADDR_ANY 0
# define ASIO_OS_DEF_MSG_OOB 0x1
# define ASIO_OS_DEF_MSG_PEEK 0x2
# define ASIO_OS_DEF_MSG_DONTROUTE 0x4
# define ASIO_OS_DEF_MSG_EOR 0 // Not supported.
# define ASIO_OS_DEF_SHUT_RD 0x0
# define ASIO_OS_DEF_SHUT_WR 0x1
# define ASIO_OS_DEF_SHUT_RDWR 0x2
# define ASIO_OS_DEF_SOMAXCONN 0x7fffffff
# define ASIO_OS_DEF_SOL_SOCKET 0xffff
# define ASIO_OS_DEF_SO_BROADCAST 0x20
# define ASIO_OS_DEF_SO_DEBUG 0x1
# define ASIO_OS_DEF_SO_DONTROUTE 0x10
# define ASIO_OS_DEF_SO_KEEPALIVE 0x8
# define ASIO_OS_DEF_SO_LINGER 0x80
# define ASIO_OS_DEF_SO_OOBINLINE 0x100
# define ASIO_OS_DEF_SO_SNDBUF 0x1001
# define ASIO_OS_DEF_SO_RCVBUF 0x1002
# define ASIO_OS_DEF_SO_SNDLOWAT 0x1003
# define ASIO_OS_DEF_SO_RCVLOWAT 0x1004
# define ASIO_OS_DEF_SO_REUSEADDR 0x4
# define ASIO_OS_DEF_TCP_NODELAY 0x1
# define ASIO_OS_DEF_IP_MULTICAST_IF 2
# define ASIO_OS_DEF_IP_MULTICAST_TTL 3
# define ASIO_OS_DEF_IP_MULTICAST_LOOP 4
# define ASIO_OS_DEF_IP_ADD_MEMBERSHIP 5
# define ASIO_OS_DEF_IP_DROP_MEMBERSHIP 6
# define ASIO_OS_DEF_IP_TTL 7
# define ASIO_OS_DEF_IPV6_UNICAST_HOPS 4
# define ASIO_OS_DEF_IPV6_MULTICAST_IF 9
# define ASIO_OS_DEF_IPV6_MULTICAST_HOPS 10
# define ASIO_OS_DEF_IPV6_MULTICAST_LOOP 11
# define ASIO_OS_DEF_IPV6_JOIN_GROUP 12
# define ASIO_OS_DEF_IPV6_LEAVE_GROUP 13
# define ASIO_OS_DEF_AI_CANONNAME 0x2
# define ASIO_OS_DEF_AI_PASSIVE 0x1
# define ASIO_OS_DEF_AI_NUMERICHOST 0x4
# define ASIO_OS_DEF_AI_NUMERICSERV 0x8
# define ASIO_OS_DEF_AI_V4MAPPED 0x800
# define ASIO_OS_DEF_AI_ALL 0x100
# define ASIO_OS_DEF_AI_ADDRCONFIG 0x400
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef SOCKET socket_type;
const SOCKET invalid_socket = INVALID_SOCKET;
const int socket_error_retval = SOCKET_ERROR;
const int max_addr_v4_str_len = 256;
const int max_addr_v6_str_len = 256;
typedef sockaddr socket_addr_type;
typedef in_addr in4_addr_type;
typedef ip_mreq in4_mreq_type;
typedef sockaddr_in sockaddr_in4_type;
# if defined(ASIO_HAS_OLD_WIN_SDK)
typedef in6_addr_emulation in6_addr_type;
typedef ipv6_mreq_emulation in6_mreq_type;
typedef sockaddr_in6_emulation sockaddr_in6_type;
typedef sockaddr_storage_emulation sockaddr_storage_type;
typedef addrinfo_emulation addrinfo_type;
# else
typedef in6_addr in6_addr_type;
typedef ipv6_mreq in6_mreq_type;
typedef sockaddr_in6 sockaddr_in6_type;
typedef sockaddr_storage sockaddr_storage_type;
typedef addrinfo addrinfo_type;
# endif
typedef ::linger linger_type;
typedef unsigned long ioctl_arg_type;
typedef u_long u_long_type;
typedef u_short u_short_type;
typedef int signed_size_type;
# define ASIO_OS_DEF(c) ASIO_OS_DEF_##c
# define ASIO_OS_DEF_AF_UNSPEC AF_UNSPEC
# define ASIO_OS_DEF_AF_INET AF_INET
# define ASIO_OS_DEF_AF_INET6 AF_INET6
# define ASIO_OS_DEF_SOCK_STREAM SOCK_STREAM
# define ASIO_OS_DEF_SOCK_DGRAM SOCK_DGRAM
# define ASIO_OS_DEF_SOCK_RAW SOCK_RAW
# define ASIO_OS_DEF_SOCK_SEQPACKET SOCK_SEQPACKET
# define ASIO_OS_DEF_IPPROTO_IP IPPROTO_IP
# define ASIO_OS_DEF_IPPROTO_IPV6 IPPROTO_IPV6
# define ASIO_OS_DEF_IPPROTO_TCP IPPROTO_TCP
# define ASIO_OS_DEF_IPPROTO_UDP IPPROTO_UDP
# define ASIO_OS_DEF_IPPROTO_ICMP IPPROTO_ICMP
# define ASIO_OS_DEF_IPPROTO_ICMPV6 IPPROTO_ICMPV6
# define ASIO_OS_DEF_FIONBIO FIONBIO
# define ASIO_OS_DEF_FIONREAD FIONREAD
# define ASIO_OS_DEF_INADDR_ANY INADDR_ANY
# define ASIO_OS_DEF_MSG_OOB MSG_OOB
# define ASIO_OS_DEF_MSG_PEEK MSG_PEEK
# define ASIO_OS_DEF_MSG_DONTROUTE MSG_DONTROUTE
# define ASIO_OS_DEF_MSG_EOR 0 // Not supported on Windows.
# define ASIO_OS_DEF_SHUT_RD SD_RECEIVE
# define ASIO_OS_DEF_SHUT_WR SD_SEND
# define ASIO_OS_DEF_SHUT_RDWR SD_BOTH
# define ASIO_OS_DEF_SOMAXCONN SOMAXCONN
# define ASIO_OS_DEF_SOL_SOCKET SOL_SOCKET
# define ASIO_OS_DEF_SO_BROADCAST SO_BROADCAST
# define ASIO_OS_DEF_SO_DEBUG SO_DEBUG
# define ASIO_OS_DEF_SO_DONTROUTE SO_DONTROUTE
# define ASIO_OS_DEF_SO_KEEPALIVE SO_KEEPALIVE
# define ASIO_OS_DEF_SO_LINGER SO_LINGER
# define ASIO_OS_DEF_SO_OOBINLINE SO_OOBINLINE
# define ASIO_OS_DEF_SO_SNDBUF SO_SNDBUF
# define ASIO_OS_DEF_SO_RCVBUF SO_RCVBUF
# define ASIO_OS_DEF_SO_SNDLOWAT SO_SNDLOWAT
# define ASIO_OS_DEF_SO_RCVLOWAT SO_RCVLOWAT
# define ASIO_OS_DEF_SO_REUSEADDR SO_REUSEADDR
# define ASIO_OS_DEF_TCP_NODELAY TCP_NODELAY
# define ASIO_OS_DEF_IP_MULTICAST_IF IP_MULTICAST_IF
# define ASIO_OS_DEF_IP_MULTICAST_TTL IP_MULTICAST_TTL
# define ASIO_OS_DEF_IP_MULTICAST_LOOP IP_MULTICAST_LOOP
# define ASIO_OS_DEF_IP_ADD_MEMBERSHIP IP_ADD_MEMBERSHIP
# define ASIO_OS_DEF_IP_DROP_MEMBERSHIP IP_DROP_MEMBERSHIP
# define ASIO_OS_DEF_IP_TTL IP_TTL
# define ASIO_OS_DEF_IPV6_UNICAST_HOPS IPV6_UNICAST_HOPS
# define ASIO_OS_DEF_IPV6_MULTICAST_IF IPV6_MULTICAST_IF
# define ASIO_OS_DEF_IPV6_MULTICAST_HOPS IPV6_MULTICAST_HOPS
# define ASIO_OS_DEF_IPV6_MULTICAST_LOOP IPV6_MULTICAST_LOOP
# define ASIO_OS_DEF_IPV6_JOIN_GROUP IPV6_JOIN_GROUP
# define ASIO_OS_DEF_IPV6_LEAVE_GROUP IPV6_LEAVE_GROUP
# define ASIO_OS_DEF_AI_CANONNAME AI_CANONNAME
# define ASIO_OS_DEF_AI_PASSIVE AI_PASSIVE
# define ASIO_OS_DEF_AI_NUMERICHOST AI_NUMERICHOST
# if defined(AI_NUMERICSERV)
#  define ASIO_OS_DEF_AI_NUMERICSERV AI_NUMERICSERV
# else
#  define ASIO_OS_DEF_AI_NUMERICSERV 0
# endif
# if defined(AI_V4MAPPED)
#  define ASIO_OS_DEF_AI_V4MAPPED AI_V4MAPPED
# else
#  define ASIO_OS_DEF_AI_V4MAPPED 0
# endif
# if defined(AI_ALL)
#  define ASIO_OS_DEF_AI_ALL AI_ALL
# else
#  define ASIO_OS_DEF_AI_ALL 0
# endif
# if defined(AI_ADDRCONFIG)
#  define ASIO_OS_DEF_AI_ADDRCONFIG AI_ADDRCONFIG
# else
#  define ASIO_OS_DEF_AI_ADDRCONFIG 0
# endif
# if defined (_WIN32_WINNT)
const int max_iov_len = 64;
# else
const int max_iov_len = 16;
# endif
#else
typedef int socket_type;
const int invalid_socket = -1;
const int socket_error_retval = -1;
const int max_addr_v4_str_len = INET_ADDRSTRLEN;
#if defined(INET6_ADDRSTRLEN)
const int max_addr_v6_str_len = INET6_ADDRSTRLEN + 1 + IF_NAMESIZE;
#else // defined(INET6_ADDRSTRLEN)
const int max_addr_v6_str_len = 256;
#endif // defined(INET6_ADDRSTRLEN)
typedef sockaddr socket_addr_type;
typedef in_addr in4_addr_type;
# if defined(__hpux)
// HP-UX doesn't provide ip_mreq when _XOPEN_SOURCE_EXTENDED is defined.
struct in4_mreq_type
{
  struct in_addr imr_multiaddr;
  struct in_addr imr_interface;
};
# else
typedef ip_mreq in4_mreq_type;
# endif
typedef sockaddr_in sockaddr_in4_type;
typedef in6_addr in6_addr_type;
typedef ipv6_mreq in6_mreq_type;
typedef sockaddr_in6 sockaddr_in6_type;
typedef sockaddr_storage sockaddr_storage_type;
typedef sockaddr_un sockaddr_un_type;
typedef addrinfo addrinfo_type;
typedef ::linger linger_type;
typedef int ioctl_arg_type;
typedef uint32_t u_long_type;
typedef uint16_t u_short_type;
#if defined(ASIO_HAS_SSIZE_T)
typedef ssize_t signed_size_type;
#else // defined(ASIO_HAS_SSIZE_T)
typedef int signed_size_type;
#endif // defined(ASIO_HAS_SSIZE_T)
# define ASIO_OS_DEF(c) ASIO_OS_DEF_##c
# define ASIO_OS_DEF_AF_UNSPEC AF_UNSPEC
# define ASIO_OS_DEF_AF_INET AF_INET
# define ASIO_OS_DEF_AF_INET6 AF_INET6
# define ASIO_OS_DEF_SOCK_STREAM SOCK_STREAM
# define ASIO_OS_DEF_SOCK_DGRAM SOCK_DGRAM
# define ASIO_OS_DEF_SOCK_RAW SOCK_RAW
# define ASIO_OS_DEF_SOCK_SEQPACKET SOCK_SEQPACKET
# define ASIO_OS_DEF_IPPROTO_IP IPPROTO_IP
# define ASIO_OS_DEF_IPPROTO_IPV6 IPPROTO_IPV6
# define ASIO_OS_DEF_IPPROTO_TCP IPPROTO_TCP
# define ASIO_OS_DEF_IPPROTO_UDP IPPROTO_UDP
# define ASIO_OS_DEF_IPPROTO_ICMP IPPROTO_ICMP
# define ASIO_OS_DEF_IPPROTO_ICMPV6 IPPROTO_ICMPV6
# define ASIO_OS_DEF_FIONBIO FIONBIO
# define ASIO_OS_DEF_FIONREAD FIONREAD
# define ASIO_OS_DEF_INADDR_ANY INADDR_ANY
# define ASIO_OS_DEF_MSG_OOB MSG_OOB
# define ASIO_OS_DEF_MSG_PEEK MSG_PEEK
# define ASIO_OS_DEF_MSG_DONTROUTE MSG_DONTROUTE
# define ASIO_OS_DEF_MSG_EOR MSG_EOR
# define ASIO_OS_DEF_SHUT_RD SHUT_RD
# define ASIO_OS_DEF_SHUT_WR SHUT_WR
# define ASIO_OS_DEF_SHUT_RDWR SHUT_RDWR
# define ASIO_OS_DEF_SOMAXCONN SOMAXCONN
# define ASIO_OS_DEF_SOL_SOCKET SOL_SOCKET
# define ASIO_OS_DEF_SO_BROADCAST SO_BROADCAST
# define ASIO_OS_DEF_SO_DEBUG SO_DEBUG
# define ASIO_OS_DEF_SO_DONTROUTE SO_DONTROUTE
# define ASIO_OS_DEF_SO_KEEPALIVE SO_KEEPALIVE
# define ASIO_OS_DEF_SO_LINGER SO_LINGER
# define ASIO_OS_DEF_SO_OOBINLINE SO_OOBINLINE
# define ASIO_OS_DEF_SO_SNDBUF SO_SNDBUF
# define ASIO_OS_DEF_SO_RCVBUF SO_RCVBUF
# define ASIO_OS_DEF_SO_SNDLOWAT SO_SNDLOWAT
# define ASIO_OS_DEF_SO_RCVLOWAT SO_RCVLOWAT
# define ASIO_OS_DEF_SO_REUSEADDR SO_REUSEADDR
# define ASIO_OS_DEF_TCP_NODELAY TCP_NODELAY
# define ASIO_OS_DEF_IP_MULTICAST_IF IP_MULTICAST_IF
# define ASIO_OS_DEF_IP_MULTICAST_TTL IP_MULTICAST_TTL
# define ASIO_OS_DEF_IP_MULTICAST_LOOP IP_MULTICAST_LOOP
# define ASIO_OS_DEF_IP_ADD_MEMBERSHIP IP_ADD_MEMBERSHIP
# define ASIO_OS_DEF_IP_DROP_MEMBERSHIP IP_DROP_MEMBERSHIP
# define ASIO_OS_DEF_IP_TTL IP_TTL
# define ASIO_OS_DEF_IPV6_UNICAST_HOPS IPV6_UNICAST_HOPS
# define ASIO_OS_DEF_IPV6_MULTICAST_IF IPV6_MULTICAST_IF
# define ASIO_OS_DEF_IPV6_MULTICAST_HOPS IPV6_MULTICAST_HOPS
# define ASIO_OS_DEF_IPV6_MULTICAST_LOOP IPV6_MULTICAST_LOOP
# define ASIO_OS_DEF_IPV6_JOIN_GROUP IPV6_JOIN_GROUP
# define ASIO_OS_DEF_IPV6_LEAVE_GROUP IPV6_LEAVE_GROUP
# define ASIO_OS_DEF_AI_CANONNAME AI_CANONNAME
# define ASIO_OS_DEF_AI_PASSIVE AI_PASSIVE
# define ASIO_OS_DEF_AI_NUMERICHOST AI_NUMERICHOST
# if defined(AI_NUMERICSERV)
#  define ASIO_OS_DEF_AI_NUMERICSERV AI_NUMERICSERV
# else
#  define ASIO_OS_DEF_AI_NUMERICSERV 0
# endif
// Note: QNX Neutrino 6.3 defines AI_V4MAPPED, AI_ALL and AI_ADDRCONFIG but
// does not implement them. Therefore they are specifically excluded here.
# if defined(AI_V4MAPPED) && !defined(__QNXNTO__)
#  define ASIO_OS_DEF_AI_V4MAPPED AI_V4MAPPED
# else
#  define ASIO_OS_DEF_AI_V4MAPPED 0
# endif
# if defined(AI_ALL) && !defined(__QNXNTO__)
#  define ASIO_OS_DEF_AI_ALL AI_ALL
# else
#  define ASIO_OS_DEF_AI_ALL 0
# endif
# if defined(AI_ADDRCONFIG) && !defined(__QNXNTO__)
#  define ASIO_OS_DEF_AI_ADDRCONFIG AI_ADDRCONFIG
# else
#  define ASIO_OS_DEF_AI_ADDRCONFIG 0
# endif
# if defined(IOV_MAX)
const int max_iov_len = IOV_MAX;
# else
// POSIX platforms are not required to define IOV_MAX.
const int max_iov_len = 16;
# endif
#endif
const int custom_socket_option_level = 0xA5100000;
const int enable_connection_aborted_option = 1;
const int always_fail_option = 2;

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_TYPES_HPP
