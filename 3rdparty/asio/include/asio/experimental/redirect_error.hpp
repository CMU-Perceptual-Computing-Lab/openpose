//
// experimental/redirect_error.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_REDIRECT_ERROR_HPP
#define ASIO_EXPERIMENTAL_REDIRECT_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

/// Completion token type used to specify that an error produced by an
/// asynchronous operation is captured to an error_code variable.
/**
 * The redirect_error_t class is used to indicate that any error_code produced
 * by an asynchronous operation is captured to a specified variable.
 */
template <typename CompletionToken>
class redirect_error_t
{
public:
  /// Constructor. 
  template <typename T>
  redirect_error_t(ASIO_MOVE_ARG(T) completion_token,
      asio::error_code& ec)
    : token_(ASIO_MOVE_CAST(T)(completion_token)),
      ec_(ec)
  {
  }

//private:
  CompletionToken token_;
  asio::error_code& ec_;
};

/// Create a completion token to capture error_code values to a variable.
template <typename CompletionToken>
inline redirect_error_t<typename decay<CompletionToken>::type> redirect_error(
    CompletionToken&& completion_token, asio::error_code& ec)
{
  return redirect_error_t<typename decay<CompletionToken>::type>(
      ASIO_MOVE_CAST(CompletionToken)(completion_token), ec);
}

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/experimental/impl/redirect_error.hpp"

#endif // ASIO_EXPERIMENTAL_REDIRECT_ERROR_HPP
