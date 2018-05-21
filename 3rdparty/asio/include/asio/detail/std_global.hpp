//
// detail/std_global.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STD_GLOBAL_HPP
#define ASIO_DETAIL_STD_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_CALL_ONCE)

#include <exception>
#include <mutex>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
struct std_global_impl
{
  // Helper function to perform initialisation.
  static void do_init()
  {
    instance_.ptr_ = new T;
  }

  // Destructor automatically cleans up the global.
  ~std_global_impl()
  {
    delete ptr_;
  }

  static std::once_flag init_once_;
  static std_global_impl instance_;
  T* ptr_;
};

template <typename T>
std::once_flag std_global_impl<T>::init_once_;

template <typename T>
std_global_impl<T> std_global_impl<T>::instance_;

template <typename T>
T& std_global()
{
  std::call_once(std_global_impl<T>::init_once_, &std_global_impl<T>::do_init);
  return *std_global_impl<T>::instance_.ptr_;
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_STD_CALL_ONCE)

#endif // ASIO_DETAIL_STD_GLOBAL_HPP
