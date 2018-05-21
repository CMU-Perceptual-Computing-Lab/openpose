//
// handler_type.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_TYPE_HPP
#define ASIO_HANDLER_TYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// (Deprecated: Use two-parameter version of async_result.) Default handler
/// type traits provided for all completion token types.
/**
 * The handler_type traits class is used for determining the concrete handler
 * type to be used for an asynchronous operation. It allows the handler type to
 * be determined at the point where the specific completion handler signature
 * is known.
 *
 * This template may be specialised for user-defined completion token types.
 */
template <typename CompletionToken, typename Signature, typename = void>
struct handler_type
{
  /// The handler type for the specific signature.
  typedef typename conditional<
    is_same<CompletionToken, typename decay<CompletionToken>::type>::value,
    decay<CompletionToken>,
    handler_type<typename decay<CompletionToken>::type, Signature>
  >::type::type type;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_HANDLER_TYPE_HPP
