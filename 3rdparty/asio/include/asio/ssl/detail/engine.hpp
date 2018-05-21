//
// ssl/detail/engine.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_ENGINE_HPP
#define ASIO_SSL_DETAIL_ENGINE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/buffer.hpp"
#include "asio/detail/static_mutex.hpp"
#include "asio/ssl/detail/openssl_types.hpp"
#include "asio/ssl/detail/verify_callback.hpp"
#include "asio/ssl/stream_base.hpp"
#include "asio/ssl/verify_mode.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class engine
{
public:
  enum want
  {
    // Returned by functions to indicate that the engine wants input. The input
    // buffer should be updated to point to the data. The engine then needs to
    // be called again to retry the operation.
    want_input_and_retry = -2,

    // Returned by functions to indicate that the engine wants to write output.
    // The output buffer points to the data to be written. The engine then
    // needs to be called again to retry the operation.
    want_output_and_retry = -1,

    // Returned by functions to indicate that the engine doesn't need input or
    // output.
    want_nothing = 0,

    // Returned by functions to indicate that the engine wants to write output.
    // The output buffer points to the data to be written. After that the
    // operation is complete, and the engine does not need to be called again.
    want_output = 1
  };

  // Construct a new engine for the specified context.
  ASIO_DECL explicit engine(SSL_CTX* context);

  // Destructor.
  ASIO_DECL ~engine();

  // Get the underlying implementation in the native type.
  ASIO_DECL SSL* native_handle();

  // Set the peer verification mode.
  ASIO_DECL asio::error_code set_verify_mode(
      verify_mode v, asio::error_code& ec);

  // Set the peer verification depth.
  ASIO_DECL asio::error_code set_verify_depth(
      int depth, asio::error_code& ec);

  // Set a peer certificate verification callback.
  ASIO_DECL asio::error_code set_verify_callback(
      verify_callback_base* callback, asio::error_code& ec);

  // Perform an SSL handshake using either SSL_connect (client-side) or
  // SSL_accept (server-side).
  ASIO_DECL want handshake(
      stream_base::handshake_type type, asio::error_code& ec);

  // Perform a graceful shutdown of the SSL session.
  ASIO_DECL want shutdown(asio::error_code& ec);

  // Write bytes to the SSL session.
  ASIO_DECL want write(const asio::const_buffer& data,
      asio::error_code& ec, std::size_t& bytes_transferred);

  // Read bytes from the SSL session.
  ASIO_DECL want read(const asio::mutable_buffer& data,
      asio::error_code& ec, std::size_t& bytes_transferred);

  // Get output data to be written to the transport.
  ASIO_DECL asio::mutable_buffer get_output(
      const asio::mutable_buffer& data);

  // Put input data that was read from the transport.
  ASIO_DECL asio::const_buffer put_input(
      const asio::const_buffer& data);

  // Map an error::eof code returned by the underlying transport according to
  // the type and state of the SSL session. Returns a const reference to the
  // error code object, suitable for passing to a completion handler.
  ASIO_DECL const asio::error_code& map_error_code(
      asio::error_code& ec) const;

private:
  // Disallow copying and assignment.
  engine(const engine&);
  engine& operator=(const engine&);

  // Callback used when the SSL implementation wants to verify a certificate.
  ASIO_DECL static int verify_callback_function(
      int preverified, X509_STORE_CTX* ctx);

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
  // The SSL_accept function may not be thread safe. This mutex is used to
  // protect all calls to the SSL_accept function.
  ASIO_DECL static asio::detail::static_mutex& accept_mutex();
#endif // (OPENSSL_VERSION_NUMBER < 0x10000000L)

  // Perform one operation. Returns >= 0 on success or error, want_read if the
  // operation needs more input, or want_write if it needs to write some output
  // before the operation can complete.
  ASIO_DECL want perform(int (engine::* op)(void*, std::size_t),
      void* data, std::size_t length, asio::error_code& ec,
      std::size_t* bytes_transferred);

  // Adapt the SSL_accept function to the signature needed for perform().
  ASIO_DECL int do_accept(void*, std::size_t);

  // Adapt the SSL_connect function to the signature needed for perform().
  ASIO_DECL int do_connect(void*, std::size_t);

  // Adapt the SSL_shutdown function to the signature needed for perform().
  ASIO_DECL int do_shutdown(void*, std::size_t);

  // Adapt the SSL_read function to the signature needed for perform().
  ASIO_DECL int do_read(void* data, std::size_t length);

  // Adapt the SSL_write function to the signature needed for perform().
  ASIO_DECL int do_write(void* data, std::size_t length);

  SSL* ssl_;
  BIO* ext_bio_;
};

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/detail/impl/engine.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_SSL_DETAIL_ENGINE_HPP
