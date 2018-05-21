//
// ts/netfwd.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TS_NETFWD_HPP
#define ASIO_TS_NETFWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CHRONO)
# include "asio/detail/chrono.hpp"
#endif // defined(ASIO_HAS_CHRONO)

#if defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/detail/date_time_fwd.hpp"
#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

#if !defined(GENERATING_DOCUMENTATION)

#include "asio/detail/push_options.hpp"

namespace asio {

class execution_context;

template <typename T, typename Executor>
class executor_binder;

template <typename Executor>
class executor_work_guard;

class system_executor;

class executor;

template <typename Executor>
class strand;

class io_context;

template <typename Clock>
struct wait_traits;

#if defined(ASIO_HAS_BOOST_DATE_TIME)

template <typename Time>
struct time_traits;

#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

#if defined(ASIO_ENABLE_OLD_SERVICES)

template <typename Clock, typename WaitTraits>
class waitable_timer_service;

#if defined(ASIO_HAS_BOOST_DATE_TIME)

template <typename TimeType, typename TimeTraits>
class deadline_timer_service;

#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#if !defined(ASIO_BASIC_WAITABLE_TIMER_FWD_DECL)
#define ASIO_BASIC_WAITABLE_TIMER_FWD_DECL

template <typename Clock,
    typename WaitTraits = asio::wait_traits<Clock>
    ASIO_SVC_TPARAM_DEF2(= waitable_timer_service<Clock, WaitTraits>)>
class basic_waitable_timer;

#endif // !defined(ASIO_BASIC_WAITABLE_TIMER_FWD_DECL)

#if defined(ASIO_HAS_CHRONO)

typedef basic_waitable_timer<chrono::system_clock> system_timer;

typedef basic_waitable_timer<chrono::steady_clock> steady_timer;

typedef basic_waitable_timer<chrono::high_resolution_clock>
  high_resolution_timer;

#endif // defined(ASIO_HAS_CHRONO)

template <class Protocol ASIO_SVC_TPARAM>
class basic_socket;

template <typename Protocol ASIO_SVC_TPARAM>
class basic_datagram_socket;

template <typename Protocol ASIO_SVC_TPARAM>
class basic_stream_socket;

template <typename Protocol ASIO_SVC_TPARAM>
class basic_socket_acceptor;

#if !defined(ASIO_BASIC_SOCKET_STREAMBUF_FWD_DECL)
#define ASIO_BASIC_SOCKET_STREAMBUF_FWD_DECL

// Forward declaration with defaulted arguments.
template <typename Protocol
    ASIO_SVC_TPARAM_DEF1(= stream_socket_service<Protocol>),
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
  || defined(GENERATING_DOCUMENTATION)
    typename Clock = boost::posix_time::ptime,
    typename WaitTraits = time_traits<Clock>
    ASIO_SVC_TPARAM1_DEF2(= deadline_timer_service<Clock, WaitTraits>)>
#else
    typename Clock = chrono::steady_clock,
    typename WaitTraits = wait_traits<Clock>
    ASIO_SVC_TPARAM1_DEF1(= steady_timer::service_type)>
#endif
class basic_socket_streambuf;

#endif // !defined(ASIO_BASIC_SOCKET_STREAMBUF_FWD_DECL)

#if !defined(ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL)
#define ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL

// Forward declaration with defaulted arguments.
template <typename Protocol
    ASIO_SVC_TPARAM_DEF1(= stream_socket_service<Protocol>),
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
  || defined(GENERATING_DOCUMENTATION)
    typename Clock = boost::posix_time::ptime,
    typename WaitTraits = time_traits<Clock>
    ASIO_SVC_TPARAM1_DEF2(= deadline_timer_service<Clock, WaitTraits>)>
#else
    typename Clock = chrono::steady_clock,
    typename WaitTraits = wait_traits<Clock>
    ASIO_SVC_TPARAM1_DEF1(= steady_timer::service_type)>
#endif
class basic_socket_iostream;

#endif // !defined(ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL)

namespace ip {

class address;

class address_v4;

class address_v6;

template <typename Address>
class basic_address_iterator;

typedef basic_address_iterator<address_v4> address_v4_iterator;

typedef basic_address_iterator<address_v6> address_v6_iterator;

template <typename Address>
class basic_address_range;

typedef basic_address_range<address_v4> address_v4_range;

typedef basic_address_range<address_v6> address_v6_range;

class network_v4;

class network_v6;

template <typename InternetProtocol>
class basic_endpoint;

template <typename InternetProtocol>
class basic_resolver_entry;

template <typename InternetProtocol>
class basic_resolver_results;

template <typename InternetProtocol ASIO_SVC_TPARAM>
class basic_resolver;

class tcp;

class udp;

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(GENERATING_DOCUMENTATION)

#endif // ASIO_TS_NETFWD_HPP
