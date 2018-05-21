//
// detail/null_signal_blocker.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP
#define ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS) \
  || defined(ASIO_WINDOWS) \
  || defined(ASIO_WINDOWS_RUNTIME) \
  || defined(__CYGWIN__) \
  || defined(__SYMBIAN32__)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_signal_blocker
  : private noncopyable
{
public:
  // Constructor blocks all signals for the calling thread.
  null_signal_blocker()
  {
  }

  // Destructor restores the previous signal mask.
  ~null_signal_blocker()
  {
  }

  // Block all signals for the calling thread.
  void block()
  {
  }

  // Restore the previous signal mask.
  void unblock()
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_HAS_THREADS)
       // || defined(ASIO_WINDOWS)
       // || defined(ASIO_WINDOWS_RUNTIME)
       // || defined(__CYGWIN__)
       // || defined(__SYMBIAN32__)

#endif // ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP
