//
// ssl/detail/impl/engine.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_IMPL_ENGINE_IPP
#define ASIO_SSL_DETAIL_IMPL_ENGINE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/ssl/detail/engine.hpp"
#include "asio/ssl/error.hpp"
#include "asio/ssl/verify_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

engine::engine(SSL_CTX* context)
  : ssl_(::SSL_new(context))
{
  if (!ssl_)
  {
    asio::error_code ec(
        static_cast<int>(::ERR_get_error()),
        asio::error::get_ssl_category());
    asio::detail::throw_error(ec, "engine");
  }

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
  accept_mutex().init();
#endif // (OPENSSL_VERSION_NUMBER < 0x10000000L)

  ::SSL_set_mode(ssl_, SSL_MODE_ENABLE_PARTIAL_WRITE);
  ::SSL_set_mode(ssl_, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
#if defined(SSL_MODE_RELEASE_BUFFERS)
  ::SSL_set_mode(ssl_, SSL_MODE_RELEASE_BUFFERS);
#endif // defined(SSL_MODE_RELEASE_BUFFERS)

  ::BIO* int_bio = 0;
  ::BIO_new_bio_pair(&int_bio, 0, &ext_bio_, 0);
  ::SSL_set_bio(ssl_, int_bio, int_bio);
}

engine::~engine()
{
  if (SSL_get_app_data(ssl_))
  {
    delete static_cast<verify_callback_base*>(SSL_get_app_data(ssl_));
    SSL_set_app_data(ssl_, 0);
  }

  ::BIO_free(ext_bio_);
  ::SSL_free(ssl_);
}

SSL* engine::native_handle()
{
  return ssl_;
}

asio::error_code engine::set_verify_mode(
    verify_mode v, asio::error_code& ec)
{
  ::SSL_set_verify(ssl_, v, ::SSL_get_verify_callback(ssl_));

  ec = asio::error_code();
  return ec;
}

asio::error_code engine::set_verify_depth(
    int depth, asio::error_code& ec)
{
  ::SSL_set_verify_depth(ssl_, depth);

  ec = asio::error_code();
  return ec;
}

asio::error_code engine::set_verify_callback(
    verify_callback_base* callback, asio::error_code& ec)
{
  if (SSL_get_app_data(ssl_))
    delete static_cast<verify_callback_base*>(SSL_get_app_data(ssl_));

  SSL_set_app_data(ssl_, callback);

  ::SSL_set_verify(ssl_, ::SSL_get_verify_mode(ssl_),
      &engine::verify_callback_function);

  ec = asio::error_code();
  return ec;
}

int engine::verify_callback_function(int preverified, X509_STORE_CTX* ctx)
{
  if (ctx)
  {
    if (SSL* ssl = static_cast<SSL*>(
          ::X509_STORE_CTX_get_ex_data(
            ctx, ::SSL_get_ex_data_X509_STORE_CTX_idx())))
    {
      if (SSL_get_app_data(ssl))
      {
        verify_callback_base* callback =
          static_cast<verify_callback_base*>(
              SSL_get_app_data(ssl));

        verify_context verify_ctx(ctx);
        return callback->call(preverified != 0, verify_ctx) ? 1 : 0;
      }
    }
  }

  return 0;
}

engine::want engine::handshake(
    stream_base::handshake_type type, asio::error_code& ec)
{
  return perform((type == asio::ssl::stream_base::client)
      ? &engine::do_connect : &engine::do_accept, 0, 0, ec, 0);
}

engine::want engine::shutdown(asio::error_code& ec)
{
  return perform(&engine::do_shutdown, 0, 0, ec, 0);
}

engine::want engine::write(const asio::const_buffer& data,
    asio::error_code& ec, std::size_t& bytes_transferred)
{
  if (data.size() == 0)
  {
    ec = asio::error_code();
    return engine::want_nothing;
  }

  return perform(&engine::do_write,
      const_cast<void*>(data.data()),
      data.size(), ec, &bytes_transferred);
}

engine::want engine::read(const asio::mutable_buffer& data,
    asio::error_code& ec, std::size_t& bytes_transferred)
{
  if (data.size() == 0)
  {
    ec = asio::error_code();
    return engine::want_nothing;
  }

  return perform(&engine::do_read, data.data(),
      data.size(), ec, &bytes_transferred);
}

asio::mutable_buffer engine::get_output(
    const asio::mutable_buffer& data)
{
  int length = ::BIO_read(ext_bio_,
      data.data(), static_cast<int>(data.size()));

  return asio::buffer(data,
      length > 0 ? static_cast<std::size_t>(length) : 0);
}

asio::const_buffer engine::put_input(
    const asio::const_buffer& data)
{
  int length = ::BIO_write(ext_bio_,
      data.data(), static_cast<int>(data.size()));

  return asio::buffer(data +
      (length > 0 ? static_cast<std::size_t>(length) : 0));
}

const asio::error_code& engine::map_error_code(
    asio::error_code& ec) const
{
  // We only want to map the error::eof code.
  if (ec != asio::error::eof)
    return ec;

  // If there's data yet to be read, it's an error.
  if (BIO_wpending(ext_bio_))
  {
    ec = asio::ssl::error::stream_truncated;
    return ec;
  }

  // SSL v2 doesn't provide a protocol-level shutdown, so an eof on the
  // underlying transport is passed through.
#if (OPENSSL_VERSION_NUMBER < 0x10100000L)
  if (SSL_version(ssl_) == SSL2_VERSION)
    return ec;
#endif // (OPENSSL_VERSION_NUMBER < 0x10100000L)

  // Otherwise, the peer should have negotiated a proper shutdown.
  if ((::SSL_get_shutdown(ssl_) & SSL_RECEIVED_SHUTDOWN) == 0)
  {
    ec = asio::ssl::error::stream_truncated;
  }

  return ec;
}

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
asio::detail::static_mutex& engine::accept_mutex()
{
  static asio::detail::static_mutex mutex = ASIO_STATIC_MUTEX_INIT;
  return mutex;
}
#endif // (OPENSSL_VERSION_NUMBER < 0x10000000L)

engine::want engine::perform(int (engine::* op)(void*, std::size_t),
    void* data, std::size_t length, asio::error_code& ec,
    std::size_t* bytes_transferred)
{
  std::size_t pending_output_before = ::BIO_ctrl_pending(ext_bio_);
  ::ERR_clear_error();
  int result = (this->*op)(data, length);
  int ssl_error = ::SSL_get_error(ssl_, result);
  int sys_error = static_cast<int>(::ERR_get_error());
  std::size_t pending_output_after = ::BIO_ctrl_pending(ext_bio_);

  if (ssl_error == SSL_ERROR_SSL)
  {
    ec = asio::error_code(sys_error,
        asio::error::get_ssl_category());
    return want_nothing;
  }

  if (ssl_error == SSL_ERROR_SYSCALL)
  {
    ec = asio::error_code(sys_error,
        asio::error::get_system_category());
    return want_nothing;
  }

  if (result > 0 && bytes_transferred)
    *bytes_transferred = static_cast<std::size_t>(result);

  if (ssl_error == SSL_ERROR_WANT_WRITE)
  {
    ec = asio::error_code();
    return want_output_and_retry;
  }
  else if (pending_output_after > pending_output_before)
  {
    ec = asio::error_code();
    return result > 0 ? want_output : want_output_and_retry;
  }
  else if (ssl_error == SSL_ERROR_WANT_READ)
  {
    ec = asio::error_code();
    return want_input_and_retry;
  }
  else if (::SSL_get_shutdown(ssl_) & SSL_RECEIVED_SHUTDOWN)
  {
    ec = asio::error::eof;
    return want_nothing;
  }
  else
  {
    ec = asio::error_code();
    return want_nothing;
  }
}

int engine::do_accept(void*, std::size_t)
{
#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
  asio::detail::static_mutex::scoped_lock lock(accept_mutex());
#endif // (OPENSSL_VERSION_NUMBER < 0x10000000L)
  return ::SSL_accept(ssl_);
}

int engine::do_connect(void*, std::size_t)
{
  return ::SSL_connect(ssl_);
}

int engine::do_shutdown(void*, std::size_t)
{
  int result = ::SSL_shutdown(ssl_);
  if (result == 0)
    result = ::SSL_shutdown(ssl_);
  return result;
}

int engine::do_read(void* data, std::size_t length)
{
  return ::SSL_read(ssl_, data,
      length < INT_MAX ? static_cast<int>(length) : INT_MAX);
}

int engine::do_write(void* data, std::size_t length)
{
  return ::SSL_write(ssl_, data,
      length < INT_MAX ? static_cast<int>(length) : INT_MAX);
}

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_IMPL_ENGINE_IPP
