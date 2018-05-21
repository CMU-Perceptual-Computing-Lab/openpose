//
// yield.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "coroutine.hpp"

#ifndef reenter
# define reenter(c) ASIO_CORO_REENTER(c)
#endif

#ifndef yield
# define yield ASIO_CORO_YIELD
#endif

#ifndef fork
# define fork ASIO_CORO_FORK
#endif
