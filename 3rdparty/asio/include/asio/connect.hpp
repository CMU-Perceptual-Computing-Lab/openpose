//
// connect.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_CONNECT_HPP
#define ASIO_CONNECT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/basic_socket.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail
{
  char (&has_iterator_helper(...))[2];

  template <typename T>
  char has_iterator_helper(T*, typename T::iterator* = 0);

  template <typename T>
  struct has_iterator_typedef
  {
    enum { value = (sizeof((has_iterator_helper)((T*)(0))) == 1) };
  };
} // namespace detail

/// Type trait used to determine whether a type is an endpoint sequence that can
/// be used with with @c connect and @c async_connect.
template <typename T>
struct is_endpoint_sequence
{
#if defined(GENERATING_DOCUMENTATION)
  /// The value member is true if the type may be used as an endpoint sequence.
  static const bool value;
#else
  enum
  {
    value = detail::has_iterator_typedef<T>::value
  };
#endif
};

/**
 * @defgroup connect asio::connect
 *
 * @brief Establishes a socket connection by trying each endpoint in a sequence.
 */
/*@{*/

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @returns The successfully connected endpoint.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @par Example
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 * asio::connect(s, r.resolve(q)); @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename EndpointSequence>
typename Protocol::endpoint connect(
    basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, the successfully connected endpoint. Otherwise, a
 * default-constructed endpoint.
 *
 * @par Example
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 * asio::error_code ec;
 * asio::connect(s, r.resolve(q), ec);
 * if (ec)
 * {
 *   // An error occurred.
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename EndpointSequence>
typename Protocol::endpoint connect(
    basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints, asio::error_code& ec,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

#if !defined(ASIO_NO_DEPRECATED)
/// (Deprecated.) Establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s, Iterator begin,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);

/// (Deprecated.) Establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, asio::error_code& ec,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);
#endif // !defined(ASIO_NO_DEPRECATED)

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @returns An iterator denoting the successfully connected endpoint.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @par Example
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::resolver::results_type e = r.resolve(q);
 * tcp::socket s(io_context);
 * asio::connect(s, e.begin(), e.end()); @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, Iterator end);

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @par Example
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::resolver::results_type e = r.resolve(q);
 * tcp::socket s(io_context);
 * asio::error_code ec;
 * asio::connect(s, e.begin(), e.end(), ec);
 * if (ec)
 * {
 *   // An error occurred.
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, Iterator end, asio::error_code& ec);

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @returns The successfully connected endpoint.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 * tcp::endpoint e = asio::connect(s,
 *     r.resolve(q), my_connect_condition());
 * std::cout << "Connected to: " << e << std::endl; @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename EndpointSequence, typename ConnectCondition>
typename Protocol::endpoint connect(
    basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints, ConnectCondition connect_condition,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, the successfully connected endpoint. Otherwise, a
 * default-constructed endpoint.
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 * asio::error_code ec;
 * tcp::endpoint e = asio::connect(s,
 *     r.resolve(q), my_connect_condition(), ec);
 * if (ec)
 * {
 *   // An error occurred.
 * }
 * else
 * {
 *   std::cout << "Connected to: " << e << std::endl;
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename EndpointSequence, typename ConnectCondition>
typename Protocol::endpoint connect(
    basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints, ConnectCondition connect_condition,
    asio::error_code& ec,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

#if !defined(ASIO_NO_DEPRECATED)
/// (Deprecated.) Establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename ConnectCondition>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, ConnectCondition connect_condition,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);

/// (Deprecated.) Establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename ConnectCondition>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s, Iterator begin,
    ConnectCondition connect_condition, asio::error_code& ec,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);
#endif // !defined(ASIO_NO_DEPRECATED)

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @returns An iterator denoting the successfully connected endpoint.
 *
 * @throws asio::system_error Thrown on failure. If the sequence is
 * empty, the associated @c error_code is asio::error::not_found.
 * Otherwise, contains the error from the last connection attempt.
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::resolver::results_type e = r.resolve(q);
 * tcp::socket s(io_context);
 * tcp::resolver::results_type::iterator i = asio::connect(
 *     s, e.begin(), e.end(), my_connect_condition());
 * std::cout << "Connected to: " << i->endpoint() << std::endl; @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename ConnectCondition>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s, Iterator begin,
    Iterator end, ConnectCondition connect_condition);

/// Establishes a socket connection by trying each endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c connect member
 * function, once for each endpoint in the sequence, until a connection is
 * successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param ec Set to indicate what error occurred, if any. If the sequence is
 * empty, set to asio::error::not_found. Otherwise, contains the error
 * from the last connection attempt.
 *
 * @returns On success, an iterator denoting the successfully connected
 * endpoint. Otherwise, the end iterator.
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::resolver::results_type e = r.resolve(q);
 * tcp::socket s(io_context);
 * asio::error_code ec;
 * tcp::resolver::results_type::iterator i = asio::connect(
 *     s, e.begin(), e.end(), my_connect_condition());
 * if (ec)
 * {
 *   // An error occurred.
 * }
 * else
 * {
 *   std::cout << "Connected to: " << i->endpoint() << std::endl;
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename ConnectCondition>
Iterator connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, Iterator end, ConnectCondition connect_condition,
    asio::error_code& ec);

/*@}*/

/**
 * @defgroup async_connect asio::async_connect
 *
 * @brief Asynchronously establishes a socket connection by trying each
 * endpoint in a sequence.
 */
/*@{*/

/// Asynchronously establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, the successfully connected endpoint.
 *   // Otherwise, a default-constructed endpoint.
 *   const typename Protocol::endpoint& endpoint
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 *
 * // ...
 *
 * r.async_resolve(q, resolve_handler);
 *
 * // ...
 *
 * void resolve_handler(
 *     const asio::error_code& ec,
 *     tcp::resolver::results_type results)
 * {
 *   if (!ec)
 *   {
 *     asio::async_connect(s, results, connect_handler);
 *   }
 * }
 *
 * // ...
 *
 * void connect_handler(
 *     const asio::error_code& ec,
 *     const tcp::endpoint& endpoint)
 * {
 *   // ...
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename EndpointSequence, typename RangeConnectHandler>
ASIO_INITFN_RESULT_TYPE(RangeConnectHandler,
    void (asio::error_code, typename Protocol::endpoint))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints,
    ASIO_MOVE_ARG(RangeConnectHandler) handler,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

#if !defined(ASIO_NO_DEPRECATED)
/// (Deprecated.) Asynchronously establishes a socket connection by trying each
/// endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, an iterator denoting the successfully
 *   // connected endpoint. Otherwise, the end iterator.
 *   Iterator iterator
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename IteratorConnectHandler>
ASIO_INITFN_RESULT_TYPE(IteratorConnectHandler,
    void (asio::error_code, Iterator))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, ASIO_MOVE_ARG(IteratorConnectHandler) handler,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);
#endif // !defined(ASIO_NO_DEPRECATED)

/// Asynchronously establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, an iterator denoting the successfully
 *   // connected endpoint. Otherwise, the end iterator.
 *   Iterator iterator
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * @code std::vector<tcp::endpoint> endpoints = ...;
 * tcp::socket s(io_context);
 * asio::async_connect(s,
 *     endpoints.begin(), endpoints.end(),
 *     connect_handler);
 *
 * // ...
 *
 * void connect_handler(
 *     const asio::error_code& ec,
 *     std::vector<tcp::endpoint>::iterator i)
 * {
 *   // ...
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM,
    typename Iterator, typename IteratorConnectHandler>
ASIO_INITFN_RESULT_TYPE(IteratorConnectHandler,
    void (asio::error_code, Iterator))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, Iterator end,
    ASIO_MOVE_ARG(IteratorConnectHandler) handler);

/// Asynchronously establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param endpoints A sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, an iterator denoting the successfully
 *   // connected endpoint. Otherwise, the end iterator.
 *   Iterator iterator
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 *
 * // ...
 *
 * r.async_resolve(q, resolve_handler);
 *
 * // ...
 *
 * void resolve_handler(
 *     const asio::error_code& ec,
 *     tcp::resolver::results_type results)
 * {
 *   if (!ec)
 *   {
 *     asio::async_connect(s, results,
 *         my_connect_condition(),
 *         connect_handler);
 *   }
 * }
 *
 * // ...
 *
 * void connect_handler(
 *     const asio::error_code& ec,
 *     const tcp::endpoint& endpoint)
 * {
 *   if (ec)
 *   {
 *     // An error occurred.
 *   }
 *   else
 *   {
 *     std::cout << "Connected to: " << endpoint << std::endl;
 *   }
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename EndpointSequence,
    typename ConnectCondition, typename RangeConnectHandler>
ASIO_INITFN_RESULT_TYPE(RangeConnectHandler,
    void (asio::error_code, typename Protocol::endpoint))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    const EndpointSequence& endpoints, ConnectCondition connect_condition,
    ASIO_MOVE_ARG(RangeConnectHandler) handler,
    typename enable_if<is_endpoint_sequence<
        EndpointSequence>::value>::type* = 0);

#if !defined(ASIO_NO_DEPRECATED)
/// (Deprecated.) Asynchronously establishes a socket connection by trying each
/// endpoint in a sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, an iterator denoting the successfully
 *   // connected endpoint. Otherwise, the end iterator.
 *   Iterator iterator
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note This overload assumes that a default constructed object of type @c
 * Iterator represents the end of the sequence. This is a valid assumption for
 * iterator types such as @c asio::ip::tcp::resolver::iterator.
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator,
    typename ConnectCondition, typename IteratorConnectHandler>
ASIO_INITFN_RESULT_TYPE(IteratorConnectHandler,
    void (asio::error_code, Iterator))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s, Iterator begin,
    ConnectCondition connect_condition,
    ASIO_MOVE_ARG(IteratorConnectHandler) handler,
    typename enable_if<!is_endpoint_sequence<Iterator>::value>::type* = 0);
#endif // !defined(ASIO_NO_DEPRECATED)

/// Asynchronously establishes a socket connection by trying each endpoint in a
/// sequence.
/**
 * This function attempts to connect a socket to one of a sequence of
 * endpoints. It does this by repeated calls to the socket's @c async_connect
 * member function, once for each endpoint in the sequence, until a connection
 * is successfully established.
 *
 * @param s The socket to be connected. If the socket is already open, it will
 * be closed.
 *
 * @param begin An iterator pointing to the start of a sequence of endpoints.
 *
 * @param end An iterator pointing to the end of a sequence of endpoints.
 *
 * @param connect_condition A function object that is called prior to each
 * connection attempt. The signature of the function object must be:
 * @code bool connect_condition(
 *     const asio::error_code& ec,
 *     const typename Protocol::endpoint& next); @endcode
 * The @c ec parameter contains the result from the most recent connect
 * operation. Before the first connection attempt, @c ec is always set to
 * indicate success. The @c next parameter is the next endpoint to be tried.
 * The function object should return true if the next endpoint should be tried,
 * and false if it should be skipped.
 *
 * @param handler The handler to be called when the connect operation
 * completes. Copies will be made of the handler as required. The function
 * signature of the handler must be:
 * @code void handler(
 *   // Result of operation. if the sequence is empty, set to
 *   // asio::error::not_found. Otherwise, contains the
 *   // error from the last connection attempt.
 *   const asio::error_code& error,
 *
 *   // On success, an iterator denoting the successfully
 *   // connected endpoint. Otherwise, the end iterator.
 *   Iterator iterator
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation
 * of the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * The following connect condition function object can be used to output
 * information about the individual connection attempts:
 * @code struct my_connect_condition
 * {
 *   bool operator()(
 *       const asio::error_code& ec,
 *       const::tcp::endpoint& next)
 *   {
 *     if (ec) std::cout << "Error: " << ec.message() << std::endl;
 *     std::cout << "Trying: " << next << std::endl;
 *     return true;
 *   }
 * }; @endcode
 * It would be used with the asio::connect function as follows:
 * @code tcp::resolver r(io_context);
 * tcp::resolver::query q("host", "service");
 * tcp::socket s(io_context);
 *
 * // ...
 *
 * r.async_resolve(q, resolve_handler);
 *
 * // ...
 *
 * void resolve_handler(
 *     const asio::error_code& ec,
 *     tcp::resolver::iterator i)
 * {
 *   if (!ec)
 *   {
 *     tcp::resolver::iterator end;
 *     asio::async_connect(s, i, end,
 *         my_connect_condition(),
 *         connect_handler);
 *   }
 * }
 *
 * // ...
 *
 * void connect_handler(
 *     const asio::error_code& ec,
 *     tcp::resolver::iterator i)
 * {
 *   if (ec)
 *   {
 *     // An error occurred.
 *   }
 *   else
 *   {
 *     std::cout << "Connected to: " << i->endpoint() << std::endl;
 *   }
 * } @endcode
 */
template <typename Protocol ASIO_SVC_TPARAM, typename Iterator,
    typename ConnectCondition, typename IteratorConnectHandler>
ASIO_INITFN_RESULT_TYPE(IteratorConnectHandler,
    void (asio::error_code, Iterator))
async_connect(basic_socket<Protocol ASIO_SVC_TARG>& s,
    Iterator begin, Iterator end, ConnectCondition connect_condition,
    ASIO_MOVE_ARG(IteratorConnectHandler) handler);

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/connect.hpp"

#endif
