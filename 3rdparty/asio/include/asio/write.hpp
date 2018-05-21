//
// write.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WRITE_HPP
#define ASIO_WRITE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/buffer.hpp"
#include "asio/error.hpp"

#if !defined(ASIO_NO_EXTENSIONS)
# include "asio/basic_streambuf_fwd.hpp"
#endif // !defined(ASIO_NO_EXTENSIONS)

#include "asio/detail/push_options.hpp"

namespace asio {

/**
 * @defgroup write asio::write
 *
 * @brief Write a certain amount of data to a stream before returning.
 */
/*@{*/

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written. The sum
 * of the buffer sizes indicates the maximum number of bytes to write to the
 * stream.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @par Example
 * To write a single data buffer use the @ref buffer function as follows:
 * @code asio::write(s, asio::buffer(data, size)); @endcode
 * See the @ref buffer documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, buffers,
 *     asio::transfer_all()); @endcode
 */
template <typename SyncWriteStream, typename ConstBufferSequence>
std::size_t write(SyncWriteStream& s, const ConstBufferSequence& buffers,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written. The sum
 * of the buffer sizes indicates the maximum number of bytes to write to the
 * stream.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes transferred.
 *
 * @par Example
 * To write a single data buffer use the @ref buffer function as follows:
 * @code asio::write(s, asio::buffer(data, size), ec); @endcode
 * See the @ref buffer documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, buffers,
 *     asio::transfer_all(), ec); @endcode
 */
template <typename SyncWriteStream, typename ConstBufferSequence>
std::size_t write(SyncWriteStream& s, const ConstBufferSequence& buffers,
    asio::error_code& ec,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written. The sum
 * of the buffer sizes indicates the maximum number of bytes to write to the
 * stream.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @par Example
 * To write a single data buffer use the @ref buffer function as follows:
 * @code asio::write(s, asio::buffer(data, size),
 *     asio::transfer_at_least(32)); @endcode
 * See the @ref buffer documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename SyncWriteStream, typename ConstBufferSequence,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written. The sum
 * of the buffer sizes indicates the maximum number of bytes to write to the
 * stream.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes written. If an error occurs, returns the total
 * number of bytes successfully transferred prior to the error.
 */
template <typename SyncWriteStream, typename ConstBufferSequence,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition, asio::error_code& ec,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Successfully written data is automatically consumed from the buffers.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, buffers,
 *     asio::transfer_all()); @endcode
 */
template <typename SyncWriteStream, typename DynamicBuffer>
std::size_t write(SyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Successfully written data is automatically consumed from the buffers.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes transferred.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, buffers,
 *     asio::transfer_all(), ec); @endcode
 */
template <typename SyncWriteStream, typename DynamicBuffer>
std::size_t write(SyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    asio::error_code& ec,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Successfully written data is automatically consumed from the buffers.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 */
template <typename SyncWriteStream, typename DynamicBuffer,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Successfully written data is automatically consumed from the buffers.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes written. If an error occurs, returns the total
 * number of bytes successfully transferred prior to the error.
 */
template <typename SyncWriteStream, typename DynamicBuffer,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition, asio::error_code& ec,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param b The basic_streambuf object from which data will be written.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, b,
 *     asio::transfer_all()); @endcode
 */
template <typename SyncWriteStream, typename Allocator>
std::size_t write(SyncWriteStream& s, basic_streambuf<Allocator>& b);

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param b The basic_streambuf object from which data will be written.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes transferred.
 *
 * @note This overload is equivalent to calling:
 * @code asio::write(
 *     s, b,
 *     asio::transfer_all(), ec); @endcode
 */
template <typename SyncWriteStream, typename Allocator>
std::size_t write(SyncWriteStream& s, basic_streambuf<Allocator>& b,
    asio::error_code& ec);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param b The basic_streambuf object from which data will be written.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @returns The number of bytes transferred.
 *
 * @throws asio::system_error Thrown on failure.
 */
template <typename SyncWriteStream, typename Allocator,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s, basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition);

/// Write a certain amount of data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * write_some function.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the SyncWriteStream concept.
 *
 * @param b The basic_streambuf object from which data will be written.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's write_some function.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes written. If an error occurs, returns the total
 * number of bytes successfully transferred prior to the error.
 */
template <typename SyncWriteStream, typename Allocator,
    typename CompletionCondition>
std::size_t write(SyncWriteStream& s, basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition, asio::error_code& ec);

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

/*@}*/
/**
 * @defgroup async_write asio::async_write
 *
 * @brief Start an asynchronous operation to write a certain amount of data to a
 * stream.
 */
/*@{*/

/// Start an asynchronous operation to write all of the supplied data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of
 * the handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * To write a single data buffer use the @ref buffer function as follows:
 * @code
 * asio::async_write(s, asio::buffer(data, size), handler);
 * @endcode
 * See the @ref buffer documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename AsyncWriteStream, typename ConstBufferSequence,
    typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s, const ConstBufferSequence& buffers,
    ASIO_MOVE_ARG(WriteHandler) handler,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Start an asynchronous operation to write a certain amount of data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest async_write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's async_write_some function.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 *
 * @par Example
 * To write a single data buffer use the @ref buffer function as follows:
 * @code asio::async_write(s,
 *     asio::buffer(data, size),
 *     asio::transfer_at_least(32),
 *     handler); @endcode
 * See the @ref buffer documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename AsyncWriteStream, typename ConstBufferSequence,
    typename CompletionCondition, typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s, const ConstBufferSequence& buffers,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(WriteHandler) handler,
    typename enable_if<
      is_const_buffer_sequence<ConstBufferSequence>::value
    >::type* = 0);

/// Start an asynchronous operation to write all of the supplied data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called. Successfully written
 * data is automatically consumed from the buffers.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 */
template <typename AsyncWriteStream,
    typename DynamicBuffer, typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    ASIO_MOVE_ARG(WriteHandler) handler,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

/// Start an asynchronous operation to write a certain amount of data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied dynamic buffer sequence has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param buffers The dynamic buffer sequence from which data will be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called. Successfully written
 * data is automatically consumed from the buffers.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest async_write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's async_write_some function.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 */
template <typename AsyncWriteStream, typename DynamicBuffer,
    typename CompletionCondition, typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s,
    ASIO_MOVE_ARG(DynamicBuffer) buffers,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(WriteHandler) handler,
    typename enable_if<
      is_dynamic_buffer<typename decay<DynamicBuffer>::type>::value
    >::type* = 0);

#if !defined(ASIO_NO_EXTENSIONS)
#if !defined(ASIO_NO_IOSTREAM)

/// Start an asynchronous operation to write all of the supplied data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param b A basic_streambuf object from which data will be written. Ownership
 * of the streambuf is retained by the caller, which must guarantee that it
 * remains valid until the handler is called.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 */
template <typename AsyncWriteStream, typename Allocator, typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s, basic_streambuf<Allocator>& b,
    ASIO_MOVE_ARG(WriteHandler) handler);

/// Start an asynchronous operation to write a certain amount of data to a
/// stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied basic_streambuf has been written.
 *
 * @li The completion_condition function object returns 0.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_write_some function, and is known as a <em>composed operation</em>. The
 * program must ensure that the stream performs no other write operations (such
 * as async_write, the stream's async_write_some function, or any other composed
 * operations that perform writes) until this operation completes.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the AsyncWriteStream concept.
 *
 * @param b A basic_streambuf object from which data will be written. Ownership
 * of the streambuf is retained by the caller, which must guarantee that it
 * remains valid until the handler is called.
 *
 * @param completion_condition The function object to be called to determine
 * whether the write operation is complete. The signature of the function object
 * must be:
 * @code std::size_t completion_condition(
 *   // Result of latest async_write_some operation.
 *   const asio::error_code& error,
 *
 *   // Number of bytes transferred so far.
 *   std::size_t bytes_transferred
 * ); @endcode
 * A return value of 0 indicates that the write operation is complete. A
 * non-zero return value indicates the maximum number of bytes to be written on
 * the next call to the stream's async_write_some function.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error, // Result of operation.
 *
 *   std::size_t bytes_transferred           // Number of bytes written from the
 *                                           // buffers. If an error occurred,
 *                                           // this will be less than the sum
 *                                           // of the buffer sizes.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_context::post().
 */
template <typename AsyncWriteStream, typename Allocator,
    typename CompletionCondition, typename WriteHandler>
ASIO_INITFN_RESULT_TYPE(WriteHandler,
    void (asio::error_code, std::size_t))
async_write(AsyncWriteStream& s, basic_streambuf<Allocator>& b,
    CompletionCondition completion_condition,
    ASIO_MOVE_ARG(WriteHandler) handler);

#endif // !defined(ASIO_NO_IOSTREAM)
#endif // !defined(ASIO_NO_EXTENSIONS)

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/write.hpp"

#endif // ASIO_WRITE_HPP
