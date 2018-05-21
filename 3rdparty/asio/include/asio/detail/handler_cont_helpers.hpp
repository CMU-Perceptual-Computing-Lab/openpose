//
// detail/handler_cont_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/handler_continuation_hook.hpp"

#include "asio/detail/push_options.hpp"

// Calls to asio_handler_is_continuation must be made from a namespace that
// does not contain overloads of this function. This namespace is defined here
// for that purpose.
namespace asio_handler_cont_helpers {

template <typename Context>
inline bool is_continuation(Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
  return false;
#else
  using asio::asio_handler_is_continuation;
  return asio_handler_is_continuation(
      asio::detail::addressof(context));
#endif
}

} // namespace asio_handler_cont_helpers

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP
