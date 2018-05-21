//
// impl/executor.ipp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_EXECUTOR_IPP
#define ASIO_IMPL_EXECUTOR_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

bad_executor::bad_executor() ASIO_NOEXCEPT
{
}

const char* bad_executor::what() const ASIO_NOEXCEPT_OR_NOTHROW
{
  return "bad executor";
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_EXECUTOR_IPP
