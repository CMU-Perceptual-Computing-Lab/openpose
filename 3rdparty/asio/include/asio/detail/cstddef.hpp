//
// detail/cstddef.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CSTDDEF_HPP
#define ASIO_DETAIL_CSTDDEF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstddef>

namespace asio {

#if defined(ASIO_HAS_NULLPTR)
using std::nullptr_t;
#else // defined(ASIO_HAS_NULLPTR)
struct nullptr_t {};
#endif // defined(ASIO_HAS_NULLPTR)

} // namespace asio

#endif // ASIO_DETAIL_CSTDDEF_HPP
