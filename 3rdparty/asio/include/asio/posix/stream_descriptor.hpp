//
// posix/stream_descriptor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_STREAM_DESCRIPTOR_HPP
#define ASIO_POSIX_STREAM_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/posix/descriptor.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
  || defined(GENERATING_DOCUMENTATION)

#if defined(ASIO_ENABLE_OLD_SERVICES)
# include "asio/posix/basic_stream_descriptor.hpp"
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

namespace asio {
namespace posix {

#if defined(ASIO_ENABLE_OLD_SERVICES)
// Typedef for the typical usage of a stream-oriented descriptor.
typedef basic_stream_descriptor<> stream_descriptor;
#else // defined(ASIO_ENABLE_OLD_SERVICES)
/// Provides stream-oriented descriptor functionality.
/**
 * The posix::stream_descriptor class template provides asynchronous and
 * blocking stream-oriented descriptor functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * AsyncReadStream, AsyncWriteStream, Stream, SyncReadStream, SyncWriteStream.
 */
class stream_descriptor
  : public descriptor
{
public:
  /// Construct a stream_descriptor without opening it.
  /**
   * This constructor creates a stream descriptor without opening it. The
   * descriptor needs to be opened and then connected or accepted before data
   * can be sent or received on it.
   *
   * @param io_context The io_context object that the stream descriptor will
   * use to dispatch handlers for any asynchronous operations performed on the
   * descriptor.
   */
  explicit stream_descriptor(asio::io_context& io_context)
    : descriptor(io_context)
  {
  }

  /// Construct a stream_descriptor on an existing native descriptor.
  /**
   * This constructor creates a stream descriptor object to hold an existing
   * native descriptor.
   *
   * @param io_context The io_context object that the stream descriptor will
   * use to dispatch handlers for any asynchronous operations performed on the
   * descriptor.
   *
   * @param native_descriptor The new underlying descriptor implementation.
   *
   * @throws asio::system_error Thrown on failure.
   */
  stream_descriptor(asio::io_context& io_context,
      const native_handle_type& native_descriptor)
    : descriptor(io_context, native_descriptor)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a stream_descriptor from another.
  /**
   * This constructor moves a stream descriptor from one object to another.
   *
   * @param other The other stream_descriptor object from which the move
   * will occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c stream_descriptor(io_context&) constructor.
   */
  stream_descriptor(stream_descriptor&& other)
    : descriptor(std::move(other))
  {
  }

  /// Move-assign a stream_descriptor from another.
  /**
   * This assignment operator moves a stream descriptor from one object to
   * another.
   *
   * @param other The other stream_descriptor object from which the move
   * will occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c stream_descriptor(io_context&) constructor.
   */
  stream_descriptor& operator=(stream_descriptor&& other)
  {
    descriptor::operator=(std::move(other));
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Write some data to the descriptor.
  /**
   * This function is used to write data to the stream descriptor. The function
   * call will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the descriptor.
   *
   * @returns The number of bytes written.
   *
   * @throws asio::system_error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   *
   * @par Example
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * descriptor.write_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename ConstBufferSequence>
  std::size_t write_some(const ConstBufferSequence& buffers)
  {
    asio::error_code ec;
    std::size_t s = this->get_service().write_some(
        this->get_implementation(), buffers, ec);
    asio::detail::throw_error(ec, "write_some");
    return s;
  }

  /// Write some data to the descriptor.
  /**
   * This function is used to write data to the stream descriptor. The function
   * call will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the descriptor.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes written. Returns 0 if an error occurred.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   */
  template <typename ConstBufferSequence>
  std::size_t write_some(const ConstBufferSequence& buffers,
      asio::error_code& ec)
  {
    return this->get_service().write_some(
        this->get_implementation(), buffers, ec);
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write data to the stream
   * descriptor. The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be written to the descriptor.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes written.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * descriptor.async_write_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename ConstBufferSequence, typename WriteHandler>
  ASIO_INITFN_RESULT_TYPE(WriteHandler,
      void (asio::error_code, std::size_t))
  async_write_some(const ConstBufferSequence& buffers,
      ASIO_MOVE_ARG(WriteHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a WriteHandler.
    ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

    asio::async_completion<WriteHandler,
      void (asio::error_code, std::size_t)> init(handler);

    this->get_service().async_write_some(
        this->get_implementation(), buffers, init.completion_handler);

    return init.result.get();
  }

  /// Read some data from the descriptor.
  /**
   * This function is used to read data from the stream descriptor. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::system_error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   *
   * @par Example
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * descriptor.read_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename MutableBufferSequence>
  std::size_t read_some(const MutableBufferSequence& buffers)
  {
    asio::error_code ec;
    std::size_t s = this->get_service().read_some(
        this->get_implementation(), buffers, ec);
    asio::detail::throw_error(ec, "read_some");
    return s;
  }

  /// Read some data from the descriptor.
  /**
   * This function is used to read data from the stream descriptor. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes read. Returns 0 if an error occurred.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   */
  template <typename MutableBufferSequence>
  std::size_t read_some(const MutableBufferSequence& buffers,
      asio::error_code& ec)
  {
    return this->get_service().read_some(
        this->get_implementation(), buffers, ec);
  }

  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream
   * descriptor. The function call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be read.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes read.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the @ref async_read function if you need to ensure that the
   * requested amount of data is read before the asynchronous operation
   * completes.
   *
   * @par Example
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * descriptor.async_read_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename MutableBufferSequence, typename ReadHandler>
  ASIO_INITFN_RESULT_TYPE(ReadHandler,
      void (asio::error_code, std::size_t))
  async_read_some(const MutableBufferSequence& buffers,
      ASIO_MOVE_ARG(ReadHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ReadHandler.
    ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

    asio::async_completion<ReadHandler,
      void (asio::error_code, std::size_t)> init(handler);

    this->get_service().async_read_some(
        this->get_implementation(), buffers, init.completion_handler);

    return init.result.get();
  }
};
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

} // namespace posix
} // namespace asio

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_POSIX_STREAM_DESCRIPTOR_HPP
