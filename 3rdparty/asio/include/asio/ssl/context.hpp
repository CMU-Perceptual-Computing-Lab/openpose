//
// ssl/context.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_CONTEXT_HPP
#define ASIO_SSL_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <string>
#include "asio/buffer.hpp"
#include "asio/io_context.hpp"
#include "asio/ssl/context_base.hpp"
#include "asio/ssl/detail/openssl_types.hpp"
#include "asio/ssl/detail/openssl_init.hpp"
#include "asio/ssl/detail/password_callback.hpp"
#include "asio/ssl/detail/verify_callback.hpp"
#include "asio/ssl/verify_mode.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {

class context
  : public context_base,
    private noncopyable
{
public:
  /// The native handle type of the SSL context.
  typedef SSL_CTX* native_handle_type;

  /// Constructor.
  ASIO_DECL explicit context(method m);

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a context from another.
  /**
   * This constructor moves an SSL context from one object to another.
   *
   * @param other The other context object from which the move will occur.
   *
   * @note Following the move, the following operations only are valid for the
   * moved-from object:
   * @li Destruction.
   * @li As a target for move-assignment.
   */
  ASIO_DECL context(context&& other);

  /// Move-assign a context from another.
  /**
   * This assignment operator moves an SSL context from one object to another.
   *
   * @param other The other context object from which the move will occur.
   *
   * @note Following the move, the following operations only are valid for the
   * moved-from object:
   * @li Destruction.
   * @li As a target for move-assignment.
   */
  ASIO_DECL context& operator=(context&& other);
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor.
  ASIO_DECL ~context();

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * context. This is intended to allow access to context functionality that is
   * not otherwise provided.
   */
  ASIO_DECL native_handle_type native_handle();

  /// Clear options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The specified options, if currently enabled on the
   * context, are cleared.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_clear_options.
   */
  ASIO_DECL void clear_options(options o);

  /// Clear options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The specified options, if currently enabled on the
   * context, are cleared.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_clear_options.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID clear_options(options o,
      asio::error_code& ec);

  /// Set options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The options are bitwise-ored with any existing
   * value for the options.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_options.
   */
  ASIO_DECL void set_options(options o);

  /// Set options on the context.
  /**
   * This function may be used to configure the SSL options used by the context.
   *
   * @param o A bitmask of options. The available option values are defined in
   * the context_base class. The options are bitwise-ored with any existing
   * value for the options.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_options.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID set_options(options o,
      asio::error_code& ec);

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the context.
   *
   * @param v A bitmask of peer verification modes. See @ref verify_mode for
   * available values.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_verify.
   */
  ASIO_DECL void set_verify_mode(verify_mode v);

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the context.
   *
   * @param v A bitmask of peer verification modes. See @ref verify_mode for
   * available values.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_verify.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID set_verify_mode(
      verify_mode v, asio::error_code& ec);

  /// Set the peer verification depth.
  /**
   * This function may be used to configure the maximum verification depth
   * allowed by the context.
   *
   * @param depth Maximum depth for the certificate chain verification that
   * shall be allowed.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_verify_depth.
   */
  ASIO_DECL void set_verify_depth(int depth);

  /// Set the peer verification depth.
  /**
   * This function may be used to configure the maximum verification depth
   * allowed by the context.
   *
   * @param depth Maximum depth for the certificate chain verification that
   * shall be allowed.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_verify_depth.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID set_verify_depth(
      int depth, asio::error_code& ec);

  /// Set the callback used to verify peer certificates.
  /**
   * This function is used to specify a callback function that will be called
   * by the implementation when it needs to verify a peer certificate.
   *
   * @param callback The function object to be used for verifying a certificate.
   * The function signature of the handler must be:
   * @code bool verify_callback(
   *   bool preverified, // True if the certificate passed pre-verification.
   *   verify_context& ctx // The peer certificate and other context.
   * ); @endcode
   * The return value of the callback is true if the certificate has passed
   * verification, false otherwise.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_verify.
   */
  template <typename VerifyCallback>
  void set_verify_callback(VerifyCallback callback);

  /// Set the callback used to verify peer certificates.
  /**
   * This function is used to specify a callback function that will be called
   * by the implementation when it needs to verify a peer certificate.
   *
   * @param callback The function object to be used for verifying a certificate.
   * The function signature of the handler must be:
   * @code bool verify_callback(
   *   bool preverified, // True if the certificate passed pre-verification.
   *   verify_context& ctx // The peer certificate and other context.
   * ); @endcode
   * The return value of the callback is true if the certificate has passed
   * verification, false otherwise.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_verify.
   */
  template <typename VerifyCallback>
  ASIO_SYNC_OP_VOID set_verify_callback(VerifyCallback callback,
      asio::error_code& ec);

  /// Load a certification authority file for performing verification.
  /**
   * This function is used to load one or more trusted certification authorities
   * from a file.
   *
   * @param filename The name of a file containing certification authority
   * certificates in PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_load_verify_locations.
   */
  ASIO_DECL void load_verify_file(const std::string& filename);

  /// Load a certification authority file for performing verification.
  /**
   * This function is used to load the certificates for one or more trusted
   * certification authorities from a file.
   *
   * @param filename The name of a file containing certification authority
   * certificates in PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_load_verify_locations.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID load_verify_file(
      const std::string& filename, asio::error_code& ec);

  /// Add certification authority for performing verification.
  /**
   * This function is used to add one trusted certification authority
   * from a memory buffer.
   *
   * @param ca The buffer containing the certification authority certificate.
   * The certificate must use the PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_get_cert_store and @c X509_STORE_add_cert.
   */
  ASIO_DECL void add_certificate_authority(const const_buffer& ca);

  /// Add certification authority for performing verification.
  /**
   * This function is used to add one trusted certification authority
   * from a memory buffer.
   *
   * @param ca The buffer containing the certification authority certificate.
   * The certificate must use the PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_get_cert_store and @c X509_STORE_add_cert.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID add_certificate_authority(
      const const_buffer& ca, asio::error_code& ec);

  /// Configures the context to use the default directories for finding
  /// certification authority certificates.
  /**
   * This function specifies that the context should use the default,
   * system-dependent directories for locating certification authority
   * certificates.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_default_verify_paths.
   */
  ASIO_DECL void set_default_verify_paths();

  /// Configures the context to use the default directories for finding
  /// certification authority certificates.
  /**
   * This function specifies that the context should use the default,
   * system-dependent directories for locating certification authority
   * certificates.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_default_verify_paths.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID set_default_verify_paths(
      asio::error_code& ec);

  /// Add a directory containing certificate authority files to be used for
  /// performing verification.
  /**
   * This function is used to specify the name of a directory containing
   * certification authority certificates. Each file in the directory must
   * contain a single certificate. The files must be named using the subject
   * name's hash and an extension of ".0".
   *
   * @param path The name of a directory containing the certificates.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_load_verify_locations.
   */
  ASIO_DECL void add_verify_path(const std::string& path);

  /// Add a directory containing certificate authority files to be used for
  /// performing verification.
  /**
   * This function is used to specify the name of a directory containing
   * certification authority certificates. Each file in the directory must
   * contain a single certificate. The files must be named using the subject
   * name's hash and an extension of ".0".
   *
   * @param path The name of a directory containing the certificates.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_load_verify_locations.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID add_verify_path(
      const std::string& path, asio::error_code& ec);

  /// Use a certificate from a memory buffer.
  /**
   * This function is used to load a certificate into the context from a buffer.
   *
   * @param certificate The buffer containing the certificate.
   *
   * @param format The certificate format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_certificate or SSL_CTX_use_certificate_ASN1.
   */
  ASIO_DECL void use_certificate(
      const const_buffer& certificate, file_format format);

  /// Use a certificate from a memory buffer.
  /**
   * This function is used to load a certificate into the context from a buffer.
   *
   * @param certificate The buffer containing the certificate.
   *
   * @param format The certificate format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_certificate or SSL_CTX_use_certificate_ASN1.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_certificate(
      const const_buffer& certificate, file_format format,
      asio::error_code& ec);

  /// Use a certificate from a file.
  /**
   * This function is used to load a certificate into the context from a file.
   *
   * @param filename The name of the file containing the certificate.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_certificate_file.
   */
  ASIO_DECL void use_certificate_file(
      const std::string& filename, file_format format);

  /// Use a certificate from a file.
  /**
   * This function is used to load a certificate into the context from a file.
   *
   * @param filename The name of the file containing the certificate.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_certificate_file.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_certificate_file(
      const std::string& filename, file_format format,
      asio::error_code& ec);

  /// Use a certificate chain from a memory buffer.
  /**
   * This function is used to load a certificate chain into the context from a
   * buffer.
   *
   * @param chain The buffer containing the certificate chain. The certificate
   * chain must use the PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_certificate and SSL_CTX_add_extra_chain_cert.
   */
  ASIO_DECL void use_certificate_chain(const const_buffer& chain);

  /// Use a certificate chain from a memory buffer.
  /**
   * This function is used to load a certificate chain into the context from a
   * buffer.
   *
   * @param chain The buffer containing the certificate chain. The certificate
   * chain must use the PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_certificate and SSL_CTX_add_extra_chain_cert.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_certificate_chain(
      const const_buffer& chain, asio::error_code& ec);

  /// Use a certificate chain from a file.
  /**
   * This function is used to load a certificate chain into the context from a
   * file.
   *
   * @param filename The name of the file containing the certificate. The file
   * must use the PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_certificate_chain_file.
   */
  ASIO_DECL void use_certificate_chain_file(const std::string& filename);

  /// Use a certificate chain from a file.
  /**
   * This function is used to load a certificate chain into the context from a
   * file.
   *
   * @param filename The name of the file containing the certificate. The file
   * must use the PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_certificate_chain_file.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_certificate_chain_file(
      const std::string& filename, asio::error_code& ec);

  /// Use a private key from a memory buffer.
  /**
   * This function is used to load a private key into the context from a buffer.
   *
   * @param private_key The buffer containing the private key.
   *
   * @param format The private key format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_PrivateKey or SSL_CTX_use_PrivateKey_ASN1.
   */
  ASIO_DECL void use_private_key(
      const const_buffer& private_key, file_format format);

  /// Use a private key from a memory buffer.
  /**
   * This function is used to load a private key into the context from a buffer.
   *
   * @param private_key The buffer containing the private key.
   *
   * @param format The private key format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_PrivateKey or SSL_CTX_use_PrivateKey_ASN1.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_private_key(
      const const_buffer& private_key, file_format format,
      asio::error_code& ec);

  /// Use a private key from a file.
  /**
   * This function is used to load a private key into the context from a file.
   *
   * @param filename The name of the file containing the private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_PrivateKey_file.
   */
  ASIO_DECL void use_private_key_file(
      const std::string& filename, file_format format);

  /// Use a private key from a file.
  /**
   * This function is used to load a private key into the context from a file.
   *
   * @param filename The name of the file containing the private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_PrivateKey_file.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_private_key_file(
      const std::string& filename, file_format format,
      asio::error_code& ec);

  /// Use an RSA private key from a memory buffer.
  /**
   * This function is used to load an RSA private key into the context from a
   * buffer.
   *
   * @param private_key The buffer containing the RSA private key.
   *
   * @param format The private key format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_RSAPrivateKey or SSL_CTX_use_RSAPrivateKey_ASN1.
   */
  ASIO_DECL void use_rsa_private_key(
      const const_buffer& private_key, file_format format);

  /// Use an RSA private key from a memory buffer.
  /**
   * This function is used to load an RSA private key into the context from a
   * buffer.
   *
   * @param private_key The buffer containing the RSA private key.
   *
   * @param format The private key format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_RSAPrivateKey or SSL_CTX_use_RSAPrivateKey_ASN1.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_rsa_private_key(
      const const_buffer& private_key, file_format format,
      asio::error_code& ec);

  /// Use an RSA private key from a file.
  /**
   * This function is used to load an RSA private key into the context from a
   * file.
   *
   * @param filename The name of the file containing the RSA private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_use_RSAPrivateKey_file.
   */
  ASIO_DECL void use_rsa_private_key_file(
      const std::string& filename, file_format format);

  /// Use an RSA private key from a file.
  /**
   * This function is used to load an RSA private key into the context from a
   * file.
   *
   * @param filename The name of the file containing the RSA private key.
   *
   * @param format The file format (ASN.1 or PEM).
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_use_RSAPrivateKey_file.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_rsa_private_key_file(
      const std::string& filename, file_format format,
      asio::error_code& ec);

  /// Use the specified memory buffer to obtain the temporary Diffie-Hellman
  /// parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a buffer.
   *
   * @param dh The memory buffer containing the Diffie-Hellman parameters. The
   * buffer must use the PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_tmp_dh.
   */
  ASIO_DECL void use_tmp_dh(const const_buffer& dh);

  /// Use the specified memory buffer to obtain the temporary Diffie-Hellman
  /// parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a buffer.
   *
   * @param dh The memory buffer containing the Diffie-Hellman parameters. The
   * buffer must use the PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_tmp_dh.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_tmp_dh(
      const const_buffer& dh, asio::error_code& ec);

  /// Use the specified file to obtain the temporary Diffie-Hellman parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a file.
   *
   * @param filename The name of the file containing the Diffie-Hellman
   * parameters. The file must use the PEM format.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_tmp_dh.
   */
  ASIO_DECL void use_tmp_dh_file(const std::string& filename);

  /// Use the specified file to obtain the temporary Diffie-Hellman parameters.
  /**
   * This function is used to load Diffie-Hellman parameters into the context
   * from a file.
   *
   * @param filename The name of the file containing the Diffie-Hellman
   * parameters. The file must use the PEM format.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_tmp_dh.
   */
  ASIO_DECL ASIO_SYNC_OP_VOID use_tmp_dh_file(
      const std::string& filename, asio::error_code& ec);

  /// Set the password callback.
  /**
   * This function is used to specify a callback function to obtain password
   * information about an encrypted key in PEM format.
   *
   * @param callback The function object to be used for obtaining the password.
   * The function signature of the handler must be:
   * @code std::string password_callback(
   *   std::size_t max_length,  // The maximum size for a password.
   *   password_purpose purpose // Whether password is for reading or writing.
   * ); @endcode
   * The return value of the callback is a string containing the password.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_CTX_set_default_passwd_cb.
   */
  template <typename PasswordCallback>
  void set_password_callback(PasswordCallback callback);

  /// Set the password callback.
  /**
   * This function is used to specify a callback function to obtain password
   * information about an encrypted key in PEM format.
   *
   * @param callback The function object to be used for obtaining the password.
   * The function signature of the handler must be:
   * @code std::string password_callback(
   *   std::size_t max_length,  // The maximum size for a password.
   *   password_purpose purpose // Whether password is for reading or writing.
   * ); @endcode
   * The return value of the callback is a string containing the password.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_CTX_set_default_passwd_cb.
   */
  template <typename PasswordCallback>
  ASIO_SYNC_OP_VOID set_password_callback(PasswordCallback callback,
      asio::error_code& ec);

private:
  struct bio_cleanup;
  struct x509_cleanup;
  struct evp_pkey_cleanup;
  struct rsa_cleanup;
  struct dh_cleanup;

  // Helper function used to set a peer certificate verification callback.
  ASIO_DECL ASIO_SYNC_OP_VOID do_set_verify_callback(
      detail::verify_callback_base* callback, asio::error_code& ec);

  // Callback used when the SSL implementation wants to verify a certificate.
  ASIO_DECL static int verify_callback_function(
      int preverified, X509_STORE_CTX* ctx);

  // Helper function used to set a password callback.
  ASIO_DECL ASIO_SYNC_OP_VOID do_set_password_callback(
      detail::password_callback_base* callback, asio::error_code& ec);

  // Callback used when the SSL implementation wants a password.
  ASIO_DECL static int password_callback_function(
      char* buf, int size, int purpose, void* data);

  // Helper function to set the temporary Diffie-Hellman parameters from a BIO.
  ASIO_DECL ASIO_SYNC_OP_VOID do_use_tmp_dh(
      BIO* bio, asio::error_code& ec);

  // Helper function to make a BIO from a memory buffer.
  ASIO_DECL BIO* make_buffer_bio(const const_buffer& b);

  // The underlying native implementation.
  native_handle_type handle_;

  // Ensure openssl is initialised.
  asio::ssl::detail::openssl_init<> init_;
};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/ssl/impl/context.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/impl/context.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_SSL_CONTEXT_HPP
