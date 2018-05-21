//
// serial_port.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Rep Invariant Systems, Inc. (info@repinvariant.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SERIAL_PORT_HPP
#define ASIO_SERIAL_PORT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_SERIAL_PORT) \
  || defined(GENERATING_DOCUMENTATION)

#include <string>
#include "asio/async_result.hpp"
#include "asio/basic_io_object.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"
#include "asio/serial_port_base.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_ENABLE_OLD_SERVICES)
# include "asio/basic_serial_port.hpp"
#else // defined(ASIO_ENABLE_OLD_SERVICES)
# if defined(ASIO_HAS_IOCP)
#  include "asio/detail/win_iocp_serial_port_service.hpp"
#  define ASIO_SVC_T detail::win_iocp_serial_port_service
# else
#  include "asio/detail/reactive_serial_port_service.hpp"
#  define ASIO_SVC_T detail::reactive_serial_port_service
# endif
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_ENABLE_OLD_SERVICES)
// Typedef for the typical usage of a serial port.
typedef basic_serial_port<> serial_port;
#else // defined(ASIO_ENABLE_OLD_SERVICES)
/// Provides serial port functionality.
/**
 * The serial_port class provides a wrapper over serial port functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class serial_port
  : ASIO_SVC_ACCESS basic_io_object<ASIO_SVC_T>,
    public serial_port_base
{
public:
  /// The type of the executor associated with the object.
  typedef io_context::executor_type executor_type;

  /// The native representation of a serial port.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef ASIO_SVC_T::native_handle_type native_handle_type;
#endif

  /// A basic_serial_port is always the lowest layer.
  typedef serial_port lowest_layer_type;

  /// Construct a serial_port without opening it.
  /**
   * This constructor creates a serial port without opening it.
   *
   * @param io_context The io_context object that the serial port will use to
   * dispatch handlers for any asynchronous operations performed on the port.
   */
  explicit serial_port(asio::io_context& io_context)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
  }

  /// Construct and open a serial_port.
  /**
   * This constructor creates and opens a serial port for the specified device
   * name.
   *
   * @param io_context The io_context object that the serial port will use to
   * dispatch handlers for any asynchronous operations performed on the port.
   *
   * @param device The platform-specific device name for this serial
   * port.
   */
  explicit serial_port(asio::io_context& io_context,
      const char* device)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
    asio::error_code ec;
    this->get_service().open(this->get_implementation(), device, ec);
    asio::detail::throw_error(ec, "open");
  }

  /// Construct and open a serial_port.
  /**
   * This constructor creates and opens a serial port for the specified device
   * name.
   *
   * @param io_context The io_context object that the serial port will use to
   * dispatch handlers for any asynchronous operations performed on the port.
   *
   * @param device The platform-specific device name for this serial
   * port.
   */
  explicit serial_port(asio::io_context& io_context,
      const std::string& device)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
    asio::error_code ec;
    this->get_service().open(this->get_implementation(), device, ec);
    asio::detail::throw_error(ec, "open");
  }

  /// Construct a serial_port on an existing native serial port.
  /**
   * This constructor creates a serial port object to hold an existing native
   * serial port.
   *
   * @param io_context The io_context object that the serial port will use to
   * dispatch handlers for any asynchronous operations performed on the port.
   *
   * @param native_serial_port A native serial port.
   *
   * @throws asio::system_error Thrown on failure.
   */
  serial_port(asio::io_context& io_context,
      const native_handle_type& native_serial_port)
    : basic_io_object<ASIO_SVC_T>(io_context)
  {
    asio::error_code ec;
    this->get_service().assign(this->get_implementation(),
        native_serial_port, ec);
    asio::detail::throw_error(ec, "assign");
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a serial_port from another.
  /**
   * This constructor moves a serial port from one object to another.
   *
   * @param other The other serial_port object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c serial_port(io_context&) constructor.
   */
  serial_port(serial_port&& other)
    : basic_io_object<ASIO_SVC_T>(std::move(other))
  {
  }

  /// Move-assign a serial_port from another.
  /**
   * This assignment operator moves a serial port from one object to another.
   *
   * @param other The other serial_port object from which the move will
   * occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c serial_port(io_context&) constructor.
   */
  serial_port& operator=(serial_port&& other)
  {
    basic_io_object<ASIO_SVC_T>::operator=(std::move(other));
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destroys the serial port.
  /**
   * This function destroys the serial port, cancelling any outstanding
   * asynchronous wait operations associated with the serial port as if by
   * calling @c cancel.
   */
  ~serial_port()
  {
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use get_executor().) Get the io_context associated with the
  /// object.
  /**
   * This function may be used to obtain the io_context object that the I/O
   * object uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the io_context object that the I/O object will use
   * to dispatch handlers. Ownership is not transferred to the caller.
   */
  asio::io_context& get_io_context()
  {
    return basic_io_object<ASIO_SVC_T>::get_io_context();
  }

  /// (Deprecated: Use get_executor().) Get the io_context associated with the
  /// object.
  /**
   * This function may be used to obtain the io_context object that the I/O
   * object uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the io_context object that the I/O object will use
   * to dispatch handlers. Ownership is not transferred to the caller.
   */
  asio::io_context& get_io_service()
  {
    return basic_io_object<ASIO_SVC_T>::get_io_service();
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Get the executor associated with the object.
  executor_type get_executor() ASIO_NOEXCEPT
  {
    return basic_io_object<ASIO_SVC_T>::get_executor();
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * layers. Since a serial_port cannot contain any further layers, it simply
   * returns a reference to itself.
   *
   * @return A reference to the lowest layer in the stack of layers. Ownership
   * is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Get a const reference to the lowest layer.
  /**
   * This function returns a const reference to the lowest layer in a stack of
   * layers. Since a serial_port cannot contain any further layers, it simply
   * returns a reference to itself.
   *
   * @return A const reference to the lowest layer in the stack of layers.
   * Ownership is not transferred to the caller.
   */
  const lowest_layer_type& lowest_layer() const
  {
    return *this;
  }

  /// Open the serial port using the specified device name.
  /**
   * This function opens the serial port for the specified device name.
   *
   * @param device The platform-specific device name.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void open(const std::string& device)
  {
    asio::error_code ec;
    this->get_service().open(this->get_implementation(), device, ec);
    asio::detail::throw_error(ec, "open");
  }

  /// Open the serial port using the specified device name.
  /**
   * This function opens the serial port using the given platform-specific
   * device name.
   *
   * @param device The platform-specific device name.
   *
   * @param ec Set the indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID open(const std::string& device,
      asio::error_code& ec)
  {
    this->get_service().open(this->get_implementation(), device, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Assign an existing native serial port to the serial port.
  /*
   * This function opens the serial port to hold an existing native serial port.
   *
   * @param native_serial_port A native serial port.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void assign(const native_handle_type& native_serial_port)
  {
    asio::error_code ec;
    this->get_service().assign(this->get_implementation(),
        native_serial_port, ec);
    asio::detail::throw_error(ec, "assign");
  }

  /// Assign an existing native serial port to the serial port.
  /*
   * This function opens the serial port to hold an existing native serial port.
   *
   * @param native_serial_port A native serial port.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID assign(const native_handle_type& native_serial_port,
      asio::error_code& ec)
  {
    this->get_service().assign(this->get_implementation(),
        native_serial_port, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Determine whether the serial port is open.
  bool is_open() const
  {
    return this->get_service().is_open(this->get_implementation());
  }

  /// Close the serial port.
  /**
   * This function is used to close the serial port. Any asynchronous read or
   * write operations will be cancelled immediately, and will complete with the
   * asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void close()
  {
    asio::error_code ec;
    this->get_service().close(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "close");
  }

  /// Close the serial port.
  /**
   * This function is used to close the serial port. Any asynchronous read or
   * write operations will be cancelled immediately, and will complete with the
   * asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID close(asio::error_code& ec)
  {
    this->get_service().close(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get the native serial port representation.
  /**
   * This function may be used to obtain the underlying representation of the
   * serial port. This is intended to allow access to native serial port
   * functionality that is not otherwise provided.
   */
  native_handle_type native_handle()
  {
    return this->get_service().native_handle(this->get_implementation());
  }

  /// Cancel all asynchronous operations associated with the serial port.
  /**
   * This function causes all outstanding asynchronous read or write operations
   * to finish immediately, and the handlers for cancelled operations will be
   * passed the asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void cancel()
  {
    asio::error_code ec;
    this->get_service().cancel(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "cancel");
  }

  /// Cancel all asynchronous operations associated with the serial port.
  /**
   * This function causes all outstanding asynchronous read or write operations
   * to finish immediately, and the handlers for cancelled operations will be
   * passed the asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
  {
    this->get_service().cancel(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Send a break sequence to the serial port.
  /**
   * This function causes a break sequence of platform-specific duration to be
   * sent out the serial port.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void send_break()
  {
    asio::error_code ec;
    this->get_service().send_break(this->get_implementation(), ec);
    asio::detail::throw_error(ec, "send_break");
  }

  /// Send a break sequence to the serial port.
  /**
   * This function causes a break sequence of platform-specific duration to be
   * sent out the serial port.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID send_break(asio::error_code& ec)
  {
    this->get_service().send_break(this->get_implementation(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Set an option on the serial port.
  /**
   * This function is used to set an option on the serial port.
   *
   * @param option The option value to be set on the serial port.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @sa SettableSerialPortOption @n
   * asio::serial_port_base::baud_rate @n
   * asio::serial_port_base::flow_control @n
   * asio::serial_port_base::parity @n
   * asio::serial_port_base::stop_bits @n
   * asio::serial_port_base::character_size
   */
  template <typename SettableSerialPortOption>
  void set_option(const SettableSerialPortOption& option)
  {
    asio::error_code ec;
    this->get_service().set_option(this->get_implementation(), option, ec);
    asio::detail::throw_error(ec, "set_option");
  }

  /// Set an option on the serial port.
  /**
   * This function is used to set an option on the serial port.
   *
   * @param option The option value to be set on the serial port.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @sa SettableSerialPortOption @n
   * asio::serial_port_base::baud_rate @n
   * asio::serial_port_base::flow_control @n
   * asio::serial_port_base::parity @n
   * asio::serial_port_base::stop_bits @n
   * asio::serial_port_base::character_size
   */
  template <typename SettableSerialPortOption>
  ASIO_SYNC_OP_VOID set_option(const SettableSerialPortOption& option,
      asio::error_code& ec)
  {
    this->get_service().set_option(this->get_implementation(), option, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Get an option from the serial port.
  /**
   * This function is used to get the current value of an option on the serial
   * port.
   *
   * @param option The option value to be obtained from the serial port.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @sa GettableSerialPortOption @n
   * asio::serial_port_base::baud_rate @n
   * asio::serial_port_base::flow_control @n
   * asio::serial_port_base::parity @n
   * asio::serial_port_base::stop_bits @n
   * asio::serial_port_base::character_size
   */
  template <typename GettableSerialPortOption>
  void get_option(GettableSerialPortOption& option)
  {
    asio::error_code ec;
    this->get_service().get_option(this->get_implementation(), option, ec);
    asio::detail::throw_error(ec, "get_option");
  }

  /// Get an option from the serial port.
  /**
   * This function is used to get the current value of an option on the serial
   * port.
   *
   * @param option The option value to be obtained from the serial port.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @sa GettableSerialPortOption @n
   * asio::serial_port_base::baud_rate @n
   * asio::serial_port_base::flow_control @n
   * asio::serial_port_base::parity @n
   * asio::serial_port_base::stop_bits @n
   * asio::serial_port_base::character_size
   */
  template <typename GettableSerialPortOption>
  ASIO_SYNC_OP_VOID get_option(GettableSerialPortOption& option,
      asio::error_code& ec)
  {
    this->get_service().get_option(this->get_implementation(), option, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Write some data to the serial port.
  /**
   * This function is used to write data to the serial port. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the serial port.
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
   * serial_port.write_some(asio::buffer(data, size));
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

  /// Write some data to the serial port.
  /**
   * This function is used to write data to the serial port. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the serial port.
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
   * This function is used to asynchronously write data to the serial port.
   * The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be written to the serial port.
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
   * serial_port.async_write_some(asio::buffer(data, size), handler);
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

    async_completion<WriteHandler,
      void (asio::error_code, std::size_t)> init(handler);

    this->get_service().async_write_some(
        this->get_implementation(), buffers, init.completion_handler);

    return init.result.get();
  }

  /// Read some data from the serial port.
  /**
   * This function is used to read data from the serial port. The function
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
   * serial_port.read_some(asio::buffer(data, size));
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

  /// Read some data from the serial port.
  /**
   * This function is used to read data from the serial port. The function
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
   * This function is used to asynchronously read data from the serial port.
   * The function call always returns immediately.
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
   * serial_port.async_read_some(asio::buffer(data, size), handler);
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

    async_completion<ReadHandler,
      void (asio::error_code, std::size_t)> init(handler);

    this->get_service().async_read_some(
        this->get_implementation(), buffers, init.completion_handler);

    return init.result.get();
  }
};
#endif // defined(ASIO_ENABLE_OLD_SERVICES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#if !defined(ASIO_ENABLE_OLD_SERVICES)
# undef ASIO_SVC_T
#endif // !defined(ASIO_ENABLE_OLD_SERVICES)

#endif // defined(ASIO_HAS_SERIAL_PORT)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_SERIAL_PORT_HPP
