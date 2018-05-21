//
// detail/handler_invoke_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_INVOKE_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_INVOKE_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/handler_invoke_hook.hpp"

#include "asio/detail/push_options.hpp"

// Calls to asio_handler_invoke must be made from a namespace that does not
// contain overloads of this function. The asio_handler_invoke_helpers
// namespace is defined here for that purpose.
namespace asio_handler_invoke_helpers {

template <typename Function, typename Context>
inline void invoke(Function& function, Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
  Function tmp(function);
  tmp();
#else
  using asio::asio_handler_invoke;
  asio_handler_invoke(function, asio::detail::addressof(context));
#endif
}

template <typename Function, typename Context>
inline void invoke(const Function& function, Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
  Function tmp(function);
  tmp();
#else
  using asio::asio_handler_invoke;
  asio_handler_invoke(function, asio::detail::addressof(context));
#endif
}

} // namespace asio_handler_invoke_helpers

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_INVOKE_HELPERS_HPP
