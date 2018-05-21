//
// detail/string_view.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STRING_VIEW_HPP
#define ASIO_DETAIL_STRING_VIEW_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STRING_VIEW)

#if defined(ASIO_HAS_STD_STRING_VIEW)
# include <string_view>
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
# include <experimental/string_view>
#else // defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
# error ASIO_HAS_STRING_VIEW is set but no string_view is available
#endif // defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)

namespace asio {

#if defined(ASIO_HAS_STD_STRING_VIEW)
using std::basic_string_view;
using std::string_view;
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
using std::experimental::basic_string_view;
using std::experimental::string_view;
#endif // defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)

} // namespace asio

# define ASIO_STRING_VIEW_PARAM asio::string_view
#else // defined(ASIO_HAS_STRING_VIEW)
# define ASIO_STRING_VIEW_PARAM const std::string&
#endif // defined(ASIO_HAS_STRING_VIEW)

#endif // ASIO_DETAIL_STRING_VIEW_HPP
