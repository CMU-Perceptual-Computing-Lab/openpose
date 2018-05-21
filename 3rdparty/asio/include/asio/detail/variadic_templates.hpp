//
// detail/variadic_templates.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_VARIADIC_TEMPLATES_HPP
#define ASIO_DETAIL_VARIADIC_TEMPLATES_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_VARIADIC_TEMPLATES)

# define ASIO_VARIADIC_TPARAMS(n) ASIO_VARIADIC_TPARAMS_##n

# define ASIO_VARIADIC_TPARAMS_1 \
  typename T1
# define ASIO_VARIADIC_TPARAMS_2 \
  typename T1, typename T2
# define ASIO_VARIADIC_TPARAMS_3 \
  typename T1, typename T2, typename T3
# define ASIO_VARIADIC_TPARAMS_4 \
  typename T1, typename T2, typename T3, typename T4
# define ASIO_VARIADIC_TPARAMS_5 \
  typename T1, typename T2, typename T3, typename T4, typename T5

# define ASIO_VARIADIC_TARGS(n) ASIO_VARIADIC_TARGS_##n

# define ASIO_VARIADIC_TARGS_1 T1
# define ASIO_VARIADIC_TARGS_2 T1, T2
# define ASIO_VARIADIC_TARGS_3 T1, T2, T3
# define ASIO_VARIADIC_TARGS_4 T1, T2, T3, T4
# define ASIO_VARIADIC_TARGS_5 T1, T2, T3, T4, T5

# define ASIO_VARIADIC_BYVAL_PARAMS(n) \
  ASIO_VARIADIC_BYVAL_PARAMS_##n

# define ASIO_VARIADIC_BYVAL_PARAMS_1 T1 x1
# define ASIO_VARIADIC_BYVAL_PARAMS_2 T1 x1, T2 x2
# define ASIO_VARIADIC_BYVAL_PARAMS_3 T1 x1, T2 x2, T3 x3
# define ASIO_VARIADIC_BYVAL_PARAMS_4 T1 x1, T2 x2, T3 x3, T4 x4
# define ASIO_VARIADIC_BYVAL_PARAMS_5 T1 x1, T2 x2, T3 x3, T4 x4, T5 x5

# define ASIO_VARIADIC_BYVAL_ARGS(n) \
  ASIO_VARIADIC_BYVAL_ARGS_##n

# define ASIO_VARIADIC_BYVAL_ARGS_1 x1
# define ASIO_VARIADIC_BYVAL_ARGS_2 x1, x2
# define ASIO_VARIADIC_BYVAL_ARGS_3 x1, x2, x3
# define ASIO_VARIADIC_BYVAL_ARGS_4 x1, x2, x3, x4
# define ASIO_VARIADIC_BYVAL_ARGS_5 x1, x2, x3, x4, x5

# define ASIO_VARIADIC_MOVE_PARAMS(n) \
  ASIO_VARIADIC_MOVE_PARAMS_##n

# define ASIO_VARIADIC_MOVE_PARAMS_1 \
  ASIO_MOVE_ARG(T1) x1
# define ASIO_VARIADIC_MOVE_PARAMS_2 \
  ASIO_MOVE_ARG(T1) x1, ASIO_MOVE_ARG(T2) x2
# define ASIO_VARIADIC_MOVE_PARAMS_3 \
  ASIO_MOVE_ARG(T1) x1, ASIO_MOVE_ARG(T2) x2, \
  ASIO_MOVE_ARG(T3) x3
# define ASIO_VARIADIC_MOVE_PARAMS_4 \
  ASIO_MOVE_ARG(T1) x1, ASIO_MOVE_ARG(T2) x2, \
  ASIO_MOVE_ARG(T3) x3, ASIO_MOVE_ARG(T4) x4
# define ASIO_VARIADIC_MOVE_PARAMS_5 \
  ASIO_MOVE_ARG(T1) x1, ASIO_MOVE_ARG(T2) x2, \
  ASIO_MOVE_ARG(T3) x3, ASIO_MOVE_ARG(T4) x4, \
  ASIO_MOVE_ARG(T5) x5

# define ASIO_VARIADIC_MOVE_ARGS(n) \
  ASIO_VARIADIC_MOVE_ARGS_##n

# define ASIO_VARIADIC_MOVE_ARGS_1 \
  ASIO_MOVE_CAST(T1)(x1)
# define ASIO_VARIADIC_MOVE_ARGS_2 \
  ASIO_MOVE_CAST(T1)(x1), ASIO_MOVE_CAST(T2)(x2)
# define ASIO_VARIADIC_MOVE_ARGS_3 \
  ASIO_MOVE_CAST(T1)(x1), ASIO_MOVE_CAST(T2)(x2), \
  ASIO_MOVE_CAST(T3)(x3)
# define ASIO_VARIADIC_MOVE_ARGS_4 \
  ASIO_MOVE_CAST(T1)(x1), ASIO_MOVE_CAST(T2)(x2), \
  ASIO_MOVE_CAST(T3)(x3), ASIO_MOVE_CAST(T4)(x4)
# define ASIO_VARIADIC_MOVE_ARGS_5 \
  ASIO_MOVE_CAST(T1)(x1), ASIO_MOVE_CAST(T2)(x2), \
  ASIO_MOVE_CAST(T3)(x3), ASIO_MOVE_CAST(T4)(x4), \
  ASIO_MOVE_CAST(T5)(x5)

# define ASIO_VARIADIC_DECAY(n) \
  ASIO_VARIADIC_DECAY_##n

# define ASIO_VARIADIC_DECAY_1 \
  typename decay<T1>::type
# define ASIO_VARIADIC_DECAY_2 \
  typename decay<T1>::type, typename decay<T2>::type
# define ASIO_VARIADIC_DECAY_3 \
  typename decay<T1>::type, typename decay<T2>::type, \
  typename decay<T3>::type
# define ASIO_VARIADIC_DECAY_4 \
  typename decay<T1>::type, typename decay<T2>::type, \
  typename decay<T3>::type, typename decay<T4>::type
# define ASIO_VARIADIC_DECAY_5 \
  typename decay<T1>::type, typename decay<T2>::type, \
  typename decay<T3>::type, typename decay<T4>::type, \
  typename decay<T5>::type

# define ASIO_VARIADIC_GENERATE(m) m(1) m(2) m(3) m(4) m(5)

#endif // !defined(ASIO_HAS_VARIADIC_TEMPLATES)

#endif // ASIO_DETAIL_VARIADIC_TEMPLATES_HPP
