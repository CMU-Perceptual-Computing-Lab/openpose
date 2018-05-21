//
// detail/keyword_tss_ptr.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_KEYWORD_TSS_PTR_HPP
#define ASIO_DETAIL_KEYWORD_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_THREAD_KEYWORD_EXTENSION)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class keyword_tss_ptr
  : private noncopyable
{
public:
  // Constructor.
  keyword_tss_ptr()
  {
  }

  // Destructor.
  ~keyword_tss_ptr()
  {
  }

  // Get the value.
  operator T*() const
  {
    return value_;
  }

  // Set the value.
  void operator=(T* value)
  {
    value_ = value;
  }

private:
  static ASIO_THREAD_KEYWORD T* value_;
};

template <typename T>
ASIO_THREAD_KEYWORD T* keyword_tss_ptr<T>::value_;

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_THREAD_KEYWORD_EXTENSION)

#endif // ASIO_DETAIL_KEYWORD_TSS_PTR_HPP
