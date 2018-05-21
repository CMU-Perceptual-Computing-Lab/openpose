//
// use_future.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_USE_FUTURE_HPP
#define ASIO_USE_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_FUTURE) \
  || defined(GENERATING_DOCUMENTATION)

#include <memory>
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Function, typename Allocator>
class packaged_token;

template <typename Function, typename Allocator, typename Result>
class packaged_handler;

} // namespace detail

/// Class used to specify that an asynchronous operation should return a future.
/**
 * The use_future_t class is used to indicate that an asynchronous operation
 * should return a std::future object. A use_future_t object may be passed as a
 * handler to an asynchronous operation, typically using the special value @c
 * asio::use_future. For example:
 *
 * @code std::future<std::size_t> my_future
 *   = my_socket.async_read_some(my_buffer, asio::use_future); @endcode
 *
 * The initiating function (async_read_some in the above example) returns a
 * future that will receive the result of the operation. If the operation
 * completes with an error_code indicating failure, it is converted into a
 * system_error and passed back to the caller via the future.
 */
template <typename Allocator = std::allocator<void> >
class use_future_t
{
public:
  /// The allocator type. The allocator is used when constructing the
  /// @c std::promise object for a given asynchronous operation.
  typedef Allocator allocator_type;

  /// Construct using default-constructed allocator.
  ASIO_CONSTEXPR use_future_t()
  {
  }

  /// Construct using specified allocator.
  explicit use_future_t(const Allocator& allocator)
    : allocator_(allocator)
  {
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use rebind().) Specify an alternate allocator.
  template <typename OtherAllocator>
  use_future_t<OtherAllocator> operator[](const OtherAllocator& allocator) const
  {
    return use_future_t<OtherAllocator>(allocator);
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Specify an alternate allocator.
  template <typename OtherAllocator>
  use_future_t<OtherAllocator> rebind(const OtherAllocator& allocator) const
  {
    return use_future_t<OtherAllocator>(allocator);
  }

  /// Obtain allocator.
  allocator_type get_allocator() const
  {
    return allocator_;
  }

  /// Wrap a function object in a packaged task.
  /**
   * The @c package function is used to adapt a function object as a packaged
   * task. When this adapter is passed as a completion token to an asynchronous
   * operation, the result of the function object is retuned via a std::future.
   *
   * @par Example
   *
   * @code std::future<std::size_t> fut =
   *   my_socket.async_read_some(buffer,
   *     use_future([](asio::error_code ec, std::size_t n)
   *       {
   *         return ec ? 0 : n;
   *       }));
   * ...
   * std::size_t n = fut.get(); @endcode
   */
  template <typename Function>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else // defined(GENERATING_DOCUMENTATION)
  detail::packaged_token<typename decay<Function>::type, Allocator>
#endif // defined(GENERATING_DOCUMENTATION)
  operator()(ASIO_MOVE_ARG(Function) f) const;

private:
  // Helper type to ensure that use_future can be constexpr default-constructed
  // even when std::allocator<void> can't be.
  struct std_allocator_void
  {
    ASIO_CONSTEXPR std_allocator_void()
    {
    }

    operator std::allocator<void>() const
    {
      return std::allocator<void>();
    }
  };

  typename conditional<
    is_same<std::allocator<void>, Allocator>::value,
    std_allocator_void, Allocator>::type allocator_;
};

/// A special value, similar to std::nothrow.
/**
 * See the documentation for asio::use_future_t for a usage example.
 */
#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr use_future_t<> use_future;
#elif defined(ASIO_MSVC)
__declspec(selectany) use_future_t<> use_future;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/use_future.hpp"

#endif // defined(ASIO_HAS_STD_FUTURE)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_USE_FUTURE_HPP
