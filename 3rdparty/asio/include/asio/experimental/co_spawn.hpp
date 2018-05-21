//
// experimental/co_spawn.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_CO_SPAWN_HPP
#define ASIO_EXPERIMENTAL_CO_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CO_AWAIT) || defined(GENERATING_DOCUMENTATION)

#include <experimental/coroutine>
#include "asio/executor.hpp"
#include "asio/strand.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

using std::experimental::coroutine_handle;

template <typename> class awaiter;
template <typename> class awaitee_base;
template <typename, typename> class awaitee;
template <typename, typename> class await_handler_base;
template <typename Executor, typename F, typename CompletionToken>
auto co_spawn(const Executor& ex, F&& f, CompletionToken&& token);

} // namespace detail

namespace this_coro {

/// Awaitable type that returns a completion token for the current coroutine.
struct token_t {};

/// Awaitable object that returns a completion token for the current coroutine.
constexpr inline token_t token() { return {}; }

/// Awaitable type that returns the executor of the current coroutine.
struct executor_t {};

/// Awaitable object that returns the executor of the current coroutine.
constexpr inline executor_t executor() { return {}; }

} // namespace this_coro

/// A completion token that represents the currently executing coroutine.
/**
 * The await_token class is used to represent the currently executing
 * coroutine. An await_token may be passed as a handler to an asynchronous
 * operation. For example:
 *
 * @code awaitable<void> my_coroutine()
 * {
 *   await_token token = co_await this_coro::token();
 *   ...
 *   std::size_t n = co_await my_socket.async_read_some(buffer, token);
 *   ...
 * } @endcode
 *
 * The initiating function (async_read_some in the above example) suspends the
 * current coroutine. The coroutine is resumed when the asynchronous operation
 * completes, and the result of the operation is returned.
 */
template <typename Executor>
class await_token
{
public:
  /// The associated executor type.
  typedef Executor executor_type;

  /// Copy constructor.
  await_token(const await_token& other) noexcept
    : awaiter_(other.awaiter_)
  {
  }

  /// Move constructor.
  await_token(await_token&& other) noexcept
    : awaiter_(std::exchange(other.awaiter_, nullptr))
  {
  }

  /// Get the associated executor.
  executor_type get_executor() const noexcept
  {
    return awaiter_->get_executor();
  }

private:
  // No assignment allowed.
  await_token& operator=(const await_token&) = delete;

  template <typename> friend class detail::awaitee_base;
  template <typename, typename> friend class detail::await_handler_base;

  // Private constructor used by awaitee_base.
  explicit await_token(detail::awaiter<Executor>* a)
    : awaiter_(a)
  {
  }

  detail::awaiter<Executor>* awaiter_;
};

/// The return type of a coroutine or asynchronous operation.
template <typename T, typename Executor = strand<executor>>
class awaitable
{
public:
  /// The type of the awaited value.
  typedef T value_type;

  /// The executor type that will be used for the coroutine.
  typedef Executor executor_type;

  /// Move constructor.
  awaitable(awaitable&& other) noexcept
    : awaitee_(std::exchange(other.awaitee_, nullptr))
  {
  }

  /// Destructor
  ~awaitable()
  {
    if (awaitee_)
    {
      detail::coroutine_handle<
        detail::awaitee<T, Executor>>::from_promise(
          *awaitee_).destroy();
    }
  }

#if !defined(GENERATING_DOCUMENTATION)

  // Support for co_await keyword.
  bool await_ready() const noexcept
  {
    return awaitee_->ready();
  }

  // Support for co_await keyword.
  void await_suspend(detail::coroutine_handle<detail::awaiter<Executor>> h)
  {
    awaitee_->attach_caller(h);
  }

  // Support for co_await keyword.
  template <class U>
  void await_suspend(detail::coroutine_handle<detail::awaitee<U, Executor>> h)
  {
    awaitee_->attach_caller(h);
  }

  // Support for co_await keyword.
  T await_resume()
  {
    return awaitee_->get();
  }

#endif // !defined(GENERATING_DOCUMENTATION)

private:
  template <typename, typename> friend class detail::awaitee;
  template <typename, typename> friend class detail::await_handler_base;

  // Not copy constructible or copy assignable.
  awaitable(const awaitable&) = delete;
  awaitable& operator=(const awaitable&) = delete;

  // Construct the awaitable from a coroutine's promise object.
  explicit awaitable(detail::awaitee<T, Executor>* a) : awaitee_(a) {}

  detail::awaitee<T, Executor>* awaitee_;
};

/// Spawn a new thread of execution.
template <typename Executor, typename F, typename CompletionToken,
    typename = typename enable_if<is_executor<Executor>::value>::type>
inline auto co_spawn(const Executor& ex, F&& f, CompletionToken&& token)
{
  return detail::co_spawn(ex, std::forward<F>(f),
      std::forward<CompletionToken>(token));
}

/// Spawn a new thread of execution.
template <typename ExecutionContext, typename F, typename CompletionToken,
    typename = typename enable_if<
      is_convertible<ExecutionContext&, execution_context&>::value>::type>
inline auto co_spawn(ExecutionContext& ctx, F&& f, CompletionToken&& token)
{
  return detail::co_spawn(ctx.get_executor(), std::forward<F>(f),
      std::forward<CompletionToken>(token));
}

/// Spawn a new thread of execution.
template <typename Executor, typename F, typename CompletionToken>
inline auto co_spawn(const await_token<Executor>& parent,
    F&& f, CompletionToken&& token)
{
  return detail::co_spawn(parent.get_executor(), std::forward<F>(f),
      std::forward<CompletionToken>(token));
}

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/experimental/impl/co_spawn.hpp"

#endif // defined(ASIO_HAS_CO_AWAIT) || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_EXPERIMENTAL_CO_SPAWN_HPP
