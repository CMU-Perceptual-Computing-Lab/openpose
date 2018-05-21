//
// read_until.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_READ_UNTIL_HPP
#define ASIO_READ_UNTIL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstddef>
#include <string>
#include "asio/async_result.hpp"
#include "asio/detail/regex_fwd.hpp"
#include "asio/detail/string_view.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"

#if !defined(ASIO_NO_EXTENSIONS)
# include "asio/basic_streambuf_fwd.hpp"
#endif // !defined(ASIO_NO_EXTENSIONS)

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail
{
  char (&has_result_type_helper(...))[2];

  template <typename T>
  char has_result_type_helper(T*, typename T::result_type* = 0);

  template <typename T>
  struct has_result_type
  {
    enum { value = (sizeof((has_result_type_helper)((T*)(0))) == 1) };
  };
} // namespace detail

/// Type trait used to determine whether a type can be used as a match condition
/// function with read_until and async_read_until.
template <typename T>
struct is_match_condition
{
#if defined(GENERATING_DOCUMENTATION)
  /// The value member is true if the type may be used as a match condition.
  static const bool value;
#else
  enum
  {
    value = asio::is_function<
        typename asio::remove_pointer<T>::type>::value
      || detail::has_result_type<T>::value
  };
#endif
};

/**
 * @defgroup read_until asio::read_until
 *
 * @brief Read data into a dynamic buffer sequence, or into a streambuf, until
 * it contains a delimiter, matches a regular expression, or a function object
 * indicates a match.
 */
/*@{*/

/// Read data into a dynamic buffer sequence until it contains a specified
/// delimiter.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains the specified
 * delimiter. The call will block until one of the following conditions is
 * true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains the delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the delimiter.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond the delimiter. An application will
 * typically leave that data in the dynamic buffer sequence for a subsequent
 * read_until operation to examine.
 *
 * @par Example
 * To read data into a @c std::string until a newline is encountered:
 * @code std::string data;
 * std::string n = asio::read_until(s,
 *     asio::dynamic_buffer(data), '\n');
 * std::string line = data.substr(0, n);
 * data.erase(0, n); @endcode
 * After the @c read_until operation completes successfully, the string @c data
 * contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the
 * delimiter, so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the buffer @c b as
 * follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers, char delim);

/// Read data into a dynamic buffer sequence until it contains a specified
/// delimiter.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains the specified
 * delimiter. The call will block until one of the following conditions is
 * true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains the delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the delimiter. Returns 0 if an error occurred.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond the delimiter. An application will
 * typically leave that data in the dynamic buffer sequence for a subsequent
 * read_until operation to examine.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    char delim, asio::error_code& ec);

/// Read data into a dynamic buffer sequence until it contains a specified
/// delimiter.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains the specified
 * delimiter. The call will block until one of the following conditions is
 * true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains the delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the delimiter.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond the delimiter. An application will
 * typically leave that data in the dynamic buffer sequence for a subsequent
 * read_until operation to examine.
 *
 * @par Example
 * To read data into a @c std::string until a CR-LF sequence is encountered:
 * @code std::string data;
 * std::string n = asio::read_until(s,
 *     asio::dynamic_buffer(data), "\r\n");
 * std::string line = data.substr(0, n);
 * data.erase(0, n); @endcode
 * After the @c read_until operation completes successfully, the string @c data
 * contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the
 * delimiter, so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the buffer @c b as
 * follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim);

/// Read data into a dynamic buffer sequence until it contains a specified
/// delimiter.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains the specified
 * delimiter. The call will block until one of the following conditions is
 * true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains the delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the delimiter. Returns 0 if an error occurred.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond the delimiter. An application will
 * typically leave that data in the dynamic buffer sequence for a subsequent
 * read_until operation to examine.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim,
    asio::error_code& ec);

#if !defined(ASIO_NO_EXTENSIONS)
#if defined(ASIO_HAS_BOOST_REGEX) \
  || defined(GENERATING_DOCUMENTATION)

/// Read data into a dynamic buffer sequence until some part of the data it
/// contains matches a regular expression.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains some data
 * that matches a regular expression. The call will block until one of the
 * following conditions is true:
 *
 * @li A substring of the dynamic buffer sequence's get area matches the
 * regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains data that matches the regular expression, the function returns
 * immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers A dynamic buffer sequence into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the substring that matches the regular expression.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond that which matched the regular
 * expression. An application will typically leave that data in the dynamic
 * buffer sequence for a subsequent read_until operation to examine.
 *
 * @par Example
 * To read data into a @c std::string until a CR-LF sequence is encountered:
 * @code std::string data;
 * std::string n = asio::read_until(s,
 *     asio::dynamic_buffer(data), boost::regex("\r\n"));
 * std::string line = data.substr(0, n);
 * data.erase(0, n); @endcode
 * After the @c read_until operation completes successfully, the string @c data
 * contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the
 * delimiter, so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the buffer @c b as
 * follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr);

/// Read data into a dynamic buffer sequence until some part of the data it
/// contains matches a regular expression.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until the dynamic buffer sequence's get area contains some data
 * that matches a regular expression. The call will block until one of the
 * following conditions is true:
 *
 * @li A substring of the dynamic buffer sequence's get area matches the
 * regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the dynamic buffer sequence's get area already
 * contains data that matches the regular expression, the function returns
 * immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers A dynamic buffer sequence into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area up to
 * and including the substring that matches the regular expression. Returns 0
 * if an error occurred.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond that which matched the regular
 * expression. An application will typically leave that data in the dynamic
 * buffer sequence for a subsequent read_until operation to examine.
 */
template <typename SyncReadStream, typename DynamicBuffer>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr, asio::error_code& ec);

#endif // defined(ASIO_HAS_BOOST_REGEX)
       // || defined(GENERATING_DOCUMENTATION)

/// Read data into a dynamic buffer sequence until a function object indicates a
/// match.

/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until a user-defined match condition function object, when applied
 * to the data contained in the dynamic buffer sequence, indicates a successful
 * match. The call will block until one of the following conditions is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the match condition function object already indicates
 * a match, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers A dynamic buffer sequence into which the data will be read.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<typename DynamicBuffer::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @returns The number of bytes in the dynamic_buffer's get area that
 * have been fully consumed by the match function.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond that which matched the function object.
 * An application will typically leave that data in the dynamic buffer sequence
 * for a subsequent read_until operation to examine.

 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 *
 * @par Examples
 * To read data into a dynamic buffer sequence until whitespace is encountered:
 * @code typedef asio::buffers_iterator<
 *     asio::const_buffers_1> iterator;
 *
 * std::pair<iterator, bool>
 * match_whitespace(iterator begin, iterator end)
 * {
 *   iterator i = begin;
 *   while (i != end)
 *     if (std::isspace(*i++))
 *       return std::make_pair(i, true);
 *   return std::make_pair(i, false);
 * }
 * ...
 * std::string data;
 * asio::read_until(s, data, match_whitespace);
 * @endcode
 *
 * To read data into a @c std::string until a matching character is found:
 * @code class match_char
 * {
 * public:
 *   explicit match_char(char c) : c_(c) {}
 *
 *   template <typename Iterator>
 *   std::pair<Iterator, bool> operator()(
 *       Iterator begin, Iterator end) const
 *   {
 *     Iterator i = begin;
 *     while (i != end)
 *       if (c_ == *i++)
 *         return std::make_pair(i, true);
 *     return std::make_pair(i, false);
 *   }
 *
 * private:
 *   char c_;
 * };
 *
 * namespace asio {
 *   template <> struct is_match_condition<match_char>
 *     : public boost::true_type {};
 * } // namespace asio
 * ...
 * std::string data;
 * asio::read_until(s, data, match_char('a'));
 * @endcode
 */
template <typename SyncReadStream,
    typename DynamicBuffer, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

/// Read data into a dynamic buffer sequence until a function object indicates a
/// match.
/**
 * This function is used to read data into the specified dynamic buffer
 * sequence until a user-defined match condition function object, when applied
 * to the data contained in the dynamic buffer sequence, indicates a successful
 * match. The call will block until one of the following conditions is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the match condition function object already indicates
 * a match, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param buffers A dynamic buffer sequence into which the data will be read.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<DynamicBuffer::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the dynamic buffer sequence's get area that
 * have been fully consumed by the match function. Returns 0 if an error
 * occurred.
 *
 * @note After a successful read_until operation, the dynamic buffer sequence
 * may contain additional data beyond that which matched the function object.
 * An application will typically leave that data in the dynamic buffer sequence
 * for a subsequent read_until operation to examine.
 *
 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 */
template <typename SyncReadStream,
    typename DynamicBuffer, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition, asio::error_code& ec,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

#if !defined(ASIO_NO_IOSTREAM)

/// Read data into a streambuf until it contains a specified delimiter.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond the delimiter. An application will typically leave
 * that data in the streambuf for a subsequent read_until operation to examine.
 *
 * @par Example
 * To read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, '\n');
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 * After the @c read_until operation completes successfully, the buffer @c b
 * contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim);

/// Read data into a streambuf until it contains a specified delimiter.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter. Returns 0 if an error occurred.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond the delimiter. An application will typically leave
 * that data in the streambuf for a subsequent read_until operation to examine.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim,
    asio::error_code& ec);

/// Read data into a streambuf until it contains a specified delimiter.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond the delimiter. An application will typically leave
 * that data in the streambuf for a subsequent read_until operation to examine.
 *
 * @par Example
 * To read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, "\r\n");
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 * After the @c read_until operation completes successfully, the buffer @c b
 * contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim);

/// Read data into a streambuf until it contains a specified delimiter.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter. Returns 0 if an error occurred.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond the delimiter. An application will typically leave
 * that data in the streambuf for a subsequent read_until operation to examine.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim, asio::error_code& ec);

#if defined(ASIO_HAS_BOOST_REGEX) \
  || defined(GENERATING_DOCUMENTATION)

/// Read data into a streambuf until some part of the data it contains matches
/// a regular expression.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains some data that matches a regular expression.
 * The call will block until one of the following conditions is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains data that
 * matches the regular expression, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the substring that matches the regular expression.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond that which matched the regular expression. An
 * application will typically leave that data in the streambuf for a subsequent
 * read_until operation to examine.
 *
 * @par Example
 * To read data into a streambuf until a CR-LF sequence is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, boost::regex("\r\n"));
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 * After the @c read_until operation completes successfully, the buffer @c b
 * contains the data which matched the regular expression:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c read_until operation.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr);

/// Read data into a streambuf until some part of the data it contains matches
/// a regular expression.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains some data that matches a regular expression.
 * The call will block until one of the following conditions is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains data that
 * matches the regular expression, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the substring that matches the regular expression. Returns 0 if an error
 * occurred.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond that which matched the regular expression. An
 * application will typically leave that data in the streambuf for a subsequent
 * read_until operation to examine.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    asio::error_code& ec);

#endif // defined(ASIO_HAS_BOOST_REGEX)
       // || defined(GENERATING_DOCUMENTATION)

/// Read data into a streambuf until a function object indicates a match.
/**
 * This function is used to read data into the specified streambuf until a
 * user-defined match condition function object, when applied to the data
 * contained in the streambuf, indicates a successful match. The call will
 * block until one of the following conditions is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the match condition function object already indicates
 * a match, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<basic_streambuf<Allocator>::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @returns The number of bytes in the streambuf's get area that have been fully
 * consumed by the match function.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond that which matched the function object. An application
 * will typically leave that data in the streambuf for a subsequent read_until
 * operation to examine.
 *
 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 *
 * @par Examples
 * To read data into a streambuf until whitespace is encountered:
 * @code typedef asio::buffers_iterator<
 *     asio::streambuf::const_buffers_type> iterator;
 *
 * std::pair<iterator, bool>
 * match_whitespace(iterator begin, iterator end)
 * {
 *   iterator i = begin;
 *   while (i != end)
 *     if (std::isspace(*i++))
 *       return std::make_pair(i, true);
 *   return std::make_pair(i, false);
 * }
 * ...
 * asio::streambuf b;
 * asio::read_until(s, b, match_whitespace);
 * @endcode
 *
 * To read data into a streambuf until a matching character is found:
 * @code class match_char
 * {
 * public:
 *   explicit match_char(char c) : c_(c) {}
 *
 *   template <typename Iterator>
 *   std::pair<Iterator, bool> operator()(
 *       Iterator begin, Iterator end) const
 *   {
 *     Iterator i = begin;
 *     while (i != end)
 *       if (c_ == *i++)
 *         return std::make_pair(i, true);
 *     return std::make_pair(i, false);
 *   }
 *
 * private:
 *   char c_;
 * };
 *
 * namespace asio {
 *   template <> struct is_match_condition<match_char>
 *     : public boost::true_type {};
 * } // namespace asio
 * ...
 * asio::streambuf b;
 * asio::read_until(s, b, match_char('a'));
 * @endcode
 */
template <typename SyncReadStream, typename Allocator, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, MatchCondition match_condition,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

/// Read data into a streambuf until a function object indicates a match.
/**
 * This function is used to read data into the specified streambuf until a
 * user-defined match condition function object, when applied to the data
 * contained in the streambuf, indicates a successful match. The call will
 * block until one of the following conditions is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the match condition function object already indicates
 * a match, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<basic_streambuf<Allocator>::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area that have been fully
 * consumed by the match function. Returns 0 if an error occurred.
 *
 * @note After a successful read_until operation, the streambuf may contain
 * additional data beyond that which matched the function object. An application
 * will typically leave that data in the streambuf for a subsequent read_until
 * operation to examine.
 *
 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 */
template <typename SyncReadStream, typename Allocator, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    MatchCondition match_condition, asio::error_code& ec,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

/*@}*/
/**
 * @defgroup async_read_until asio::async_read_until
 *
 * @brief Start an asynchronous operation to read data into a dynamic buffer
 * sequence, or into a streambuf, until it contains a delimiter, matches a
 * regular expression, or a function object indicates a match.
 */
/*@{*/

/// Start an asynchronous operation to read data into a dynamic buffer sequence
/// until it contains a specified delimiter.
/**
 * This function is used to asynchronously read data into the specified dynamic
 * buffer sequence until the dynamic buffer sequence's get area contains the
 * specified delimiter. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the dynamic buffer sequence's get area already contains the delimiter, this
 * asynchronous operation completes immediately. The program must ensure that
 * the stream performs no other read operations (such as async_read,
 * async_read_until, the stream's async_read_some function, or any other
 * composed operations that perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param delim The delimiter character.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the dynamic buffer sequence's
 *   // get area up to and including the delimiter.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the dynamic buffer
 * sequence may contain additional data beyond the delimiter. An application
 * will typically leave that data in the dynamic buffer sequence for a
 * subsequent async_read_until operation to examine.
 *
 * @par Example
 * To asynchronously read data into a @c std::string until a newline is
 * encountered:
 * @code std::string data;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::string line = data.substr(0, n);
 *     data.erase(0, n);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, data, '\n', handler); @endcode
 * After the @c async_read_until operation completes successfully, the buffer
 * @c data contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the
 * delimiter, so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the buffer @c data
 * as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    char delim, ASIO_MOVE_ARG(ReadHandler) handler);

/// Start an asynchronous operation to read data into a dynamic buffer sequence
/// until it contains a specified delimiter.
/**
 * This function is used to asynchronously read data into the specified dynamic
 * buffer sequence until the dynamic buffer sequence's get area contains the
 * specified delimiter. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li The get area of the dynamic buffer sequence contains the specified
 * delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the dynamic buffer sequence's get area already contains the delimiter, this
 * asynchronous operation completes immediately. The program must ensure that
 * the stream performs no other read operations (such as async_read,
 * async_read_until, the stream's async_read_some function, or any other
 * composed operations that perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param delim The delimiter string.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the dynamic buffer sequence's
 *   // get area up to and including the delimiter.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the dynamic buffer
 * sequence may contain additional data beyond the delimiter. An application
 * will typically leave that data in the dynamic buffer sequence for a
 * subsequent async_read_until operation to examine.
 *
 * @par Example
 * To asynchronously read data into a @c std::string until a CR-LF sequence is
 * encountered:
 * @code std::string data;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::string line = data.substr(0, n);
 *     data.erase(0, n);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, data, "\r\n", handler); @endcode
 * After the @c async_read_until operation completes successfully, the string
 * @c data contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the
 * delimiter, so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the string @c data
 * as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_STRING_VIEW_PARAM delim,
    ASIO_MOVE_ARG(ReadHandler) handler);

#if !defined(ASIO_NO_EXTENSIONS)
#if defined(ASIO_HAS_BOOST_REGEX) \
  || defined(GENERATING_DOCUMENTATION)

/// Start an asynchronous operation to read data into a dynamic buffer sequence
/// until some part of its data matches a regular expression.
/**
 * This function is used to asynchronously read data into the specified dynamic
 * buffer sequence until the dynamic buffer sequence's get area contains some
 * data that matches a regular expression. The function call always returns
 * immediately. The asynchronous operation will continue until one of the
 * following conditions is true:
 *
 * @li A substring of the dynamic buffer sequence's get area matches the regular
 * expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the dynamic buffer sequence's get area already contains data that matches
 * the regular expression, this asynchronous operation completes immediately.
 * The program must ensure that the stream performs no other read operations
 * (such as async_read, async_read_until, the stream's async_read_some
 * function, or any other composed operations that perform reads) until this
 * operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param expr The regular expression.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the dynamic buffer
 *   // sequence's get area up to and including the
 *   // substring that matches the regular expression.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the dynamic buffer
 * sequence may contain additional data beyond that which matched the regular
 * expression. An application will typically leave that data in the dynamic
 * buffer sequence for a subsequent async_read_until operation to examine.
 *
 * @par Example
 * To asynchronously read data into a @c std::string until a CR-LF sequence is
 * encountered:
 * @code std::string data;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::string line = data.substr(0, n);
 *     data.erase(0, n);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, data,
 *     boost::regex("\r\n"), handler); @endcode
 * After the @c async_read_until operation completes successfully, the string
 * @c data contains the data which matched the regular expression:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c substr then extracts the data up to and including the match,
 * so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r', '\n' } @endcode
 * After the call to @c erase, the remaining data is left in the string @c data
 * as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream,
    typename DynamicBuffer, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    const boost::regex& expr,
    ASIO_MOVE_ARG(ReadHandler) handler);

#endif // defined(ASIO_HAS_BOOST_REGEX)
       // || defined(GENERATING_DOCUMENTATION)

/// Start an asynchronous operation to read data into a dynamic buffer sequence
/// until a function object indicates a match.
/**
 * This function is used to asynchronously read data into the specified dynamic
 * buffer sequence until a user-defined match condition function object, when
 * applied to the data contained in the dynamic buffer sequence, indicates a
 * successful match. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the match condition function object already indicates a match, this
 * asynchronous operation completes immediately. The program must ensure that
 * the stream performs no other read operations (such as async_read,
 * async_read_until, the stream's async_read_some function, or any other
 * composed operations that perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param buffers The dynamic buffer sequence into which the data will be read.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<typename DynamicBuffer::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the dynamic buffer sequence's
 *   // get area that have been fully consumed by the match
 *   // function. O if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the dynamic buffer
 * sequence may contain additional data beyond that which matched the function
 * object. An application will typically leave that data in the dynamic buffer
 * sequence for a subsequent async_read_until operation to examine.
 *
 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 *
 * @par Examples
 * To asynchronously read data into a @c std::string until whitespace is
 * encountered:
 * @code typedef asio::buffers_iterator<
 *     asio::const_buffers_1> iterator;
 *
 * std::pair<iterator, bool>
 * match_whitespace(iterator begin, iterator end)
 * {
 *   iterator i = begin;
 *   while (i != end)
 *     if (std::isspace(*i++))
 *       return std::make_pair(i, true);
 *   return std::make_pair(i, false);
 * }
 * ...
 * void handler(const asio::error_code& e, std::size_t size);
 * ...
 * std::string data;
 * asio::async_read_until(s, data, match_whitespace, handler);
 * @endcode
 *
 * To asynchronously read data into a @c std::string until a matching character
 * is found:
 * @code class match_char
 * {
 * public:
 *   explicit match_char(char c) : c_(c) {}
 *
 *   template <typename Iterator>
 *   std::pair<Iterator, bool> operator()(
 *       Iterator begin, Iterator end) const
 *   {
 *     Iterator i = begin;
 *     while (i != end)
 *       if (c_ == *i++)
 *         return std::make_pair(i, true);
 *     return std::make_pair(i, false);
 *   }
 *
 * private:
 *   char c_;
 * };
 *
 * namespace asio {
 *   template <> struct is_match_condition<match_char>
 *     : public boost::true_type {};
 * } // namespace asio
 * ...
 * void handler(const asio::error_code& e, std::size_t size);
 * ...
 * std::string data;
 * asio::async_read_until(s, data, match_char('a'), handler);
 * @endcode
 */
template <typename AsyncReadStream, typename DynamicBuffer,
    typename MatchCondition, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    MatchCondition match_condition, ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

#if !defined(ASIO_NO_IOSTREAM)

/// Start an asynchronous operation to read data into a streambuf until it
/// contains a specified delimiter.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains the specified delimiter.
 * The function call always returns immediately. The asynchronous operation
 * will continue until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the streambuf's get area already contains the delimiter, this asynchronous
 * operation completes immediately. The program must ensure that the stream
 * performs no other read operations (such as async_read, async_read_until, the
 * stream's async_read_some function, or any other composed operations that
 * perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param delim The delimiter character.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the streambuf's get
 *   // area up to and including the delimiter.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the streambuf may
 * contain additional data beyond the delimiter. An application will typically
 * leave that data in the streambuf for a subsequent async_read_until operation
 * to examine.
 *
 * @par Example
 * To asynchronously read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, '\n', handler); @endcode
 * After the @c async_read_until operation completes successfully, the buffer
 * @c b contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    char delim, ASIO_MOVE_ARG(ReadHandler) handler);

/// Start an asynchronous operation to read data into a streambuf until it
/// contains a specified delimiter.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains the specified delimiter.
 * The function call always returns immediately. The asynchronous operation
 * will continue until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the streambuf's get area already contains the delimiter, this asynchronous
 * operation completes immediately. The program must ensure that the stream
 * performs no other read operations (such as async_read, async_read_until, the
 * stream's async_read_some function, or any other composed operations that
 * perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param delim The delimiter string.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the streambuf's get
 *   // area up to and including the delimiter.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the streambuf may
 * contain additional data beyond the delimiter. An application will typically
 * leave that data in the streambuf for a subsequent async_read_until operation
 * to examine.
 *
 * @par Example
 * To asynchronously read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, "\r\n", handler); @endcode
 * After the @c async_read_until operation completes successfully, the buffer
 * @c b contains the delimiter:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    ASIO_STRING_VIEW_PARAM delim,
    ASIO_MOVE_ARG(ReadHandler) handler);

#if defined(ASIO_HAS_BOOST_REGEX) \
  || defined(GENERATING_DOCUMENTATION)

/// Start an asynchronous operation to read data into a streambuf until some
/// part of its data matches a regular expression.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains some data that matches a
 * regular expression. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the streambuf's get area already contains data that matches the regular
 * expression, this asynchronous operation completes immediately. The program
 * must ensure that the stream performs no other read operations (such as
 * async_read, async_read_until, the stream's async_read_some function, or any
 * other composed operations that perform reads) until this operation
 * completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param expr The regular expression.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the streambuf's get
 *   // area up to and including the substring
 *   // that matches the regular. expression.
 *   // 0 if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the streambuf may
 * contain additional data beyond that which matched the regular expression. An
 * application will typically leave that data in the streambuf for a subsequent
 * async_read_until operation to examine.
 *
 * @par Example
 * To asynchronously read data into a streambuf until a CR-LF sequence is
 * encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, boost::regex("\r\n"), handler); @endcode
 * After the @c async_read_until operation completes successfully, the buffer
 * @c b contains the data which matched the regular expression:
 * @code { 'a', 'b', ..., 'c', '\r', '\n', 'd', 'e', ... } @endcode
 * The call to @c std::getline then extracts the data up to and including the
 * newline (which is discarded), so that the string @c line contains:
 * @code { 'a', 'b', ..., 'c', '\r' } @endcode
 * The remaining data is left in the buffer @c b as follows:
 * @code { 'd', 'e', ... } @endcode
 * This data may be the start of a new line, to be extracted by a subsequent
 * @c async_read_until operation.
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    ASIO_MOVE_ARG(ReadHandler) handler);

#endif // defined(ASIO_HAS_BOOST_REGEX)
       // || defined(GENERATING_DOCUMENTATION)

/// Start an asynchronous operation to read data into a streambuf until a
/// function object indicates a match.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until a user-defined match condition function object, when applied
 * to the data contained in the streambuf, indicates a successful match. The
 * function call always returns immediately. The asynchronous operation will
 * continue until one of the following conditions is true:
 *
 * @li The match condition function object returns a std::pair where the second
 * element evaluates to true.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function, and is known as a <em>composed operation</em>. If
 * the match condition function object already indicates a match, this
 * asynchronous operation completes immediately. The program must ensure that
 * the stream performs no other read operations (such as async_read,
 * async_read_until, the stream's async_read_some function, or any other
 * composed operations that perform reads) until this operation completes.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param match_condition The function object to be called to determine whether
 * a match exists. The signature of the function object must be:
 * @code pair<iterator, bool> match_condition(iterator begin, iterator end);
 * @endcode
 * where @c iterator represents the type:
 * @code buffers_iterator<basic_streambuf<Allocator>::const_buffers_type>
 * @endcode
 * The iterator parameters @c begin and @c end define the range of bytes to be
 * scanned to determine whether there is a match. The @c first member of the
 * return value is an iterator marking one-past-the-end of the bytes that have
 * been consumed by the match function. This iterator is used to calculate the
 * @c begin parameter for any subsequent invocation of the match condition. The
 * @c second member of the return value is true if a match has been found, false
 * otherwise.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   // Result of operation.
 *   const asio::error_code& error,
 *
 *   // The number of bytes in the streambuf's get
 *   // area that have been fully consumed by the
 *   // match function. O if an error occurred.
 *   std::size_t bytes_transferred
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @note After a successful async_read_until operation, the streambuf may
 * contain additional data beyond that which matched the function object. An
 * application will typically leave that data in the streambuf for a subsequent
 * async_read_until operation to examine.
 *
 * @note The default implementation of the @c is_match_condition type trait
 * evaluates to true for function pointers and function objects with a
 * @c result_type typedef. It must be specialised for other user-defined
 * function objects.
 *
 * @par Examples
 * To asynchronously read data into a streambuf until whitespace is encountered:
 * @code typedef asio::buffers_iterator<
 *     asio::streambuf::const_buffers_type> iterator;
 *
 * std::pair<iterator, bool>
 * match_whitespace(iterator begin, iterator end)
 * {
 *   iterator i = begin;
 *   while (i != end)
 *     if (std::isspace(*i++))
 *       return std::make_pair(i, true);
 *   return std::make_pair(i, false);
 * }
 * ...
 * void handler(const asio::error_code& e, std::size_t size);
 * ...
 * asio::streambuf b;
 * asio::async_read_until(s, b, match_whitespace, handler);
 * @endcode
 *
 * To asynchronously read data into a streambuf until a matching character is
 * found:
 * @code class match_char
 * {
 * public:
 *   explicit match_char(char c) : c_(c) {}
 *
 *   template <typename Iterator>
 *   std::pair<Iterator, bool> operator()(
 *       Iterator begin, Iterator end) const
 *   {
 *     Iterator i = begin;
 *     while (i != end)
 *       if (c_ == *i++)
 *         return std::make_pair(i, true);
 *     return std::make_pair(i, false);
 *   }
 *
 * private:
 *   char c_;
 * };
 *
 * namespace asio {
 *   template <> struct is_match_condition<match_char>
 *     : public boost::true_type {};
 * } // namespace asio
 * ...
 * void handler(const asio::error_code& e, std::size_t size);
 * ...
 * asio::streambuf b;
 * asio::async_read_until(s, b, match_char('a'), handler);
 * @endcode
 */
template <typename AsyncReadStream, typename Allocator,
    typename MatchCondition, typename ReadHandler>
ASIO_INITFN_RESULT_TYPE(ReadHandler,
    void (asio::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    MatchCondition match_condition, ASIO_MOVE_ARG(ReadHandler) handler,
    typename enable_if<is_match_condition<MatchCondition>::value>::type* = 0);

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/read_until.hpp"

#endif // ASIO_READ_UNTIL_HPP
