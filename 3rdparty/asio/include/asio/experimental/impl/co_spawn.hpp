//
// experimental/impl/co_spawn.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_IMPL_CO_SPAWN_HPP
#define ASIO_EXPERIMENTAL_IMPL_CO_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <tuple>
#include <utility>
#include "asio/async_result.hpp"
#include "asio/detail/thread_context.hpp"
#include "asio/detail/thread_info_base.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/dispatch.hpp"
#include "asio/post.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

// Promise object for coroutine at top of thread-of-execution "stack".
template <typename Executor>
class awaiter
{
public:
  struct deleter
  {
    void operator()(awaiter* a)
    {
      if (a)
        a->release();
    }
  };

  typedef std::unique_ptr<awaiter, deleter> ptr;

  typedef Executor executor_type;

  ~awaiter()
  {
    if (has_executor_)
      static_cast<Executor*>(static_cast<void*>(executor_))->~Executor();
  }

  void set_executor(const Executor& ex)
  {
    new (&executor_) Executor(ex);
    has_executor_ = true;
  }

  executor_type get_executor() const noexcept
  {
    return *static_cast<const Executor*>(static_cast<const void*>(executor_));
  }

  awaiter* get_return_object()
  {
    return this;
  }

  auto initial_suspend()
  {
    return std::experimental::suspend_always();
  }

  auto final_suspend()
  {
    return std::experimental::suspend_always();
  }

  void return_void()
  {
  }

  awaiter* add_ref()
  {
    ++ref_count_;
    return this;
  }

  void release()
  {
    if (--ref_count_ == 0)
      coroutine_handle<awaiter>::from_promise(*this).destroy();
  }

  void unhandled_exception()
  {
    pending_exception_ = std::current_exception();
  }

  void rethrow_unhandled_exception()
  {
    if (pending_exception_)
    {
      std::exception_ptr ex = std::exchange(pending_exception_, nullptr);
      std::rethrow_exception(ex);
    }
  }

private:
  std::size_t ref_count_ = 0;
  std::exception_ptr pending_exception_ = nullptr;
  alignas(Executor) unsigned char executor_[sizeof(Executor)];
  bool has_executor_ = false;
};

// Base promise for coroutines further down the thread-of-execution "stack".
template <typename Executor>
class awaitee_base
{
public:
#if !defined(ASIO_DISABLE_AWAITEE_RECYCLING)
  void* operator new(std::size_t size)
  {
    return asio::detail::thread_info_base::allocate(
        asio::detail::thread_info_base::awaitee_tag(),
        asio::detail::thread_context::thread_call_stack::top(),
        size);
  }

  void operator delete(void* pointer, std::size_t size)
  {
    asio::detail::thread_info_base::deallocate(
        asio::detail::thread_info_base::awaitee_tag(),
        asio::detail::thread_context::thread_call_stack::top(),
        pointer, size);
  }
#endif // !defined(ASIO_DISABLE_AWAITEE_RECYCLING)

  auto initial_suspend()
  {
    return std::experimental::suspend_never();
  }

  struct final_suspender
  {
    awaitee_base* this_;

    bool await_ready() const noexcept
    {
      return false;
    }

    void await_suspend(coroutine_handle<void>)
    {
      this_->wake_caller();
    }

    void await_resume() const noexcept
    {
    }
  };

  auto final_suspend()
  {
    return final_suspender{this};
  }

  void set_except(std::exception_ptr e)
  {
    pending_exception_ = e;
  }

  void unhandled_exception()
  {
    set_except(std::current_exception());
  }

  void rethrow_exception()
  {
    if (pending_exception_)
    {
      std::exception_ptr ex = std::exchange(pending_exception_, nullptr);
      std::rethrow_exception(ex);
    }
  }

  awaiter<Executor>* top()
  {
    return awaiter_;
  }

  coroutine_handle<void> caller()
  {
    return caller_;
  }

  bool ready() const
  {
    return ready_;
  }

  void wake_caller()
  {
    if (caller_)
      caller_.resume();
    else
      ready_ = true;
  }

  class awaitable_executor
  {
  public:
    explicit awaitable_executor(awaitee_base* a)
      : this_(a)
    {
    }

    bool await_ready() const noexcept
    {
      return this_->awaiter_ != nullptr;
    }

    template <typename U, typename Ex>
    void await_suspend(coroutine_handle<detail::awaitee<U, Ex>> h) noexcept
    {
      this_->resume_on_attach_ = h;
    }

    Executor await_resume()
    {
      return this_->awaiter_->get_executor();
    }

  private:
    awaitee_base* this_;
  };

  awaitable_executor await_transform(this_coro::executor_t) noexcept
  {
    return awaitable_executor(this);
  }

  class awaitable_token
  {
  public:
    explicit awaitable_token(awaitee_base* a)
      : this_(a)
    {
    }

    bool await_ready() const noexcept
    {
      return this_->awaiter_ != nullptr;
    }

    template <typename U, typename Ex>
    void await_suspend(coroutine_handle<detail::awaitee<U, Ex>> h) noexcept
    {
      this_->resume_on_attach_ = h;
    }

    await_token<Executor> await_resume()
    {
      return await_token<Executor>(this_->awaiter_);
    }

  private:
    awaitee_base* this_;
  };

  awaitable_token await_transform(this_coro::token_t) noexcept
  {
    return awaitable_token(this);
  }

  template <typename T>
  awaitable<T, Executor> await_transform(awaitable<T, Executor>& t) const
  {
    return std::move(t);
  }

  template <typename T>
  awaitable<T, Executor> await_transform(awaitable<T, Executor>&& t) const
  {
    return std::move(t);
  }

  std::experimental::suspend_always await_transform(
      std::experimental::suspend_always) const
  {
    return std::experimental::suspend_always();
  }

  void attach_caller(coroutine_handle<awaiter<Executor>> h)
  {
    this->caller_ = h;
    this->attach_callees(&h.promise());
  }

  template <typename U>
  void attach_caller(coroutine_handle<awaitee<U, Executor>> h)
  {
    this->caller_ = h;
    if (h.promise().awaiter_)
      this->attach_callees(h.promise().awaiter_);
    else
      h.promise().unattached_callee_ = this;
  }

  void attach_callees(awaiter<Executor>* a)
  {
    for (awaitee_base* curr = this; curr != nullptr;
        curr = std::exchange(curr->unattached_callee_, nullptr))
    {
      curr->awaiter_ = a;
      if (curr->resume_on_attach_)
        return std::exchange(curr->resume_on_attach_, nullptr).resume();
    }
  }

protected:
  awaiter<Executor>* awaiter_ = nullptr;
  coroutine_handle<void> caller_ = nullptr;
  awaitee_base<Executor>* unattached_callee_ = nullptr;
  std::exception_ptr pending_exception_ = nullptr;
  coroutine_handle<void> resume_on_attach_ = nullptr;
  bool ready_ = false;
};

// Promise object for coroutines further down the thread-of-execution "stack".
template <typename T, typename Executor>
class awaitee
  : public awaitee_base<Executor>
{
public:
  awaitee()
  {
  }

  awaitee(awaitee&& other) noexcept
    : awaitee_base<Executor>(std::move(other))
  {
  }

  ~awaitee()
  {
    if (has_result_)
      static_cast<T*>(static_cast<void*>(result_))->~T();
  }

  awaitable<T, Executor> get_return_object()
  {
    return awaitable<T, Executor>(this);
  };

  template <typename U>
  void return_value(U&& u)
  {
    new (&result_) T(std::forward<U>(u));
    has_result_ = true;
  }

  T get()
  {
    this->caller_ = nullptr;
    this->rethrow_exception();
    return std::move(*static_cast<T*>(static_cast<void*>(result_)));
  }

private:
  alignas(T) unsigned char result_[sizeof(T)];
  bool has_result_ = false;
};

// Promise object for coroutines further down the thread-of-execution "stack".
template <typename Executor>
class awaitee<void, Executor>
  : public awaitee_base<Executor>
{
public:
  awaitable<void, Executor> get_return_object()
  {
    return awaitable<void, Executor>(this);
  };

  void return_void()
  {
  }

  void get()
  {
    this->caller_ = nullptr;
    this->rethrow_exception();
  }
};

template <typename Executor>
class awaiter_task
{
public:
  typedef Executor executor_type;

  awaiter_task(awaiter<Executor>* a)
    : awaiter_(a->add_ref())
  {
  }

  awaiter_task(awaiter_task&& other) noexcept
    : awaiter_(std::exchange(other.awaiter_, nullptr))
  {
  }

  ~awaiter_task()
  {
    if (awaiter_)
    {
      // Coroutine "stack unwinding" must be performed through the executor.
      executor_type ex(awaiter_->get_executor());
      (post)(ex,
          [a = std::move(awaiter_)]() mutable
          {
            typename awaiter<Executor>::ptr(std::move(a));
          });
    }
  }

  executor_type get_executor() const noexcept
  {
    return awaiter_->get_executor();
  }

protected:
  typename awaiter<Executor>::ptr awaiter_;
};

template <typename Executor>
class co_spawn_handler : public awaiter_task<Executor>
{
public:
  using awaiter_task<Executor>::awaiter_task;

  void operator()()
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    coroutine_handle<awaiter<Executor>>::from_promise(*ptr.get()).resume();
  }
};

template <typename Executor, typename T>
class await_handler_base : public awaiter_task<Executor>
{
public:
  typedef awaitable<T, Executor> awaitable_type;

  await_handler_base(await_token<Executor> token)
    : awaiter_task<Executor>(token.awaiter_),
      awaitee_(nullptr)
  {
  }

  await_handler_base(await_handler_base&& other) noexcept
    : awaiter_task<Executor>(std::move(other)),
      awaitee_(std::exchange(other.awaitee_, nullptr))
  {
  }

  void attach_awaitee(const awaitable<T, Executor>& a)
  {
    awaitee_ = a.awaitee_;
  }

protected:
  awaitee<T, Executor>* awaitee_;
};

template <typename, typename...> class await_handler;

template <typename Executor>
class await_handler<Executor, void>
  : public await_handler_base<Executor, void>
{
public:
  using await_handler_base<Executor, void>::await_handler_base;

  void operator()()
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    this->awaitee_->return_void();
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor>
class await_handler<Executor, asio::error_code>
  : public await_handler_base<Executor, void>
{
public:
  typedef void return_type;

  using await_handler_base<Executor, void>::await_handler_base;

  void operator()(const asio::error_code& ec)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ec)
    {
      this->awaitee_->set_except(
          std::make_exception_ptr(asio::system_error(ec)));
    }
    else
      this->awaitee_->return_void();
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor>
class await_handler<Executor, std::exception_ptr>
  : public await_handler_base<Executor, void>
{
public:
  using await_handler_base<Executor, void>::await_handler_base;

  void operator()(std::exception_ptr ex)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ex)
      this->awaitee_->set_except(ex);
    else
      this->awaitee_->return_void();
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base<Executor, T>::await_handler_base;

  template <typename Arg>
  void operator()(Arg&& arg)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    this->awaitee_->return_value(std::forward<Arg>(arg));
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, asio::error_code, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base<Executor, T>::await_handler_base;

  template <typename Arg>
  void operator()(const asio::error_code& ec, Arg&& arg)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ec)
    {
      this->awaitee_->set_except(
          std::make_exception_ptr(asio::system_error(ec)));
    }
    else
      this->awaitee_->return_value(std::forward<Arg>(arg));
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename T>
class await_handler<Executor, std::exception_ptr, T>
  : public await_handler_base<Executor, T>
{
public:
  using await_handler_base<Executor, T>::await_handler_base;

  template <typename Arg>
  void operator()(std::exception_ptr ex, Arg&& arg)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ex)
      this->awaitee_->set_except(ex);
    else
      this->awaitee_->return_value(std::forward<Arg>(arg));
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename... Ts>
class await_handler
  : public await_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using await_handler_base<Executor, std::tuple<Ts...>>::await_handler_base;

  template <typename... Args>
  void operator()(Args&&... args)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    this->awaitee_->return_value(
        std::forward_as_tuple(std::forward<Args>(args)...));
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename... Ts>
class await_handler<Executor, asio::error_code, Ts...>
  : public await_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using await_handler_base<Executor, std::tuple<Ts...>>::await_handler_base;

  template <typename... Args>
  void operator()(const asio::error_code& ec, Args&&... args)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ec)
    {
      this->awaitee_->set_except(
          std::make_exception_ptr(asio::system_error(ec)));
    }
    else
    {
      this->awaitee_->return_value(
          std::forward_as_tuple(std::forward<Args>(args)...));
    }
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename Executor, typename... Ts>
class await_handler<Executor, std::exception_ptr, Ts...>
  : public await_handler_base<Executor, std::tuple<Ts...>>
{
public:
  using await_handler_base<Executor, std::tuple<Ts...>>::await_handler_base;

  template <typename... Args>
  void operator()(std::exception_ptr ex, Args&&... args)
  {
    typename awaiter<Executor>::ptr ptr(std::move(this->awaiter_));
    if (ex)
      this->awaitee_->set_except(ex);
    else
    {
      this->awaitee_->return_value(
          std::forward_as_tuple(std::forward<Args>(args)...));
    }
    this->awaitee_->wake_caller();
    ptr->rethrow_unhandled_exception();
  }
};

template <typename T>
struct awaitable_signature;

template <typename T, typename Executor>
struct awaitable_signature<awaitable<T, Executor>>
{
  typedef void type(std::exception_ptr, T);
};

template <typename Executor>
struct awaitable_signature<awaitable<void, Executor>>
{
  typedef void type(std::exception_ptr);
};

template <typename T, typename Executor, typename F, typename Handler>
awaiter<Executor>* co_spawn_entry_point(awaitable<T, Executor>*,
    executor_work_guard<Executor> work_guard, F f, Handler handler)
{
  bool done = false;

  try
  {
    T t = co_await f();

    done = true;

    (dispatch)(work_guard.get_executor(),
        [handler = std::move(handler), t = std::move(t)]() mutable
        {
          handler(std::exception_ptr(), std::move(t));
        });
  }
  catch (...)
  {
    if (done)
      throw;

    (dispatch)(work_guard.get_executor(),
        [handler = std::move(handler), e = std::current_exception()]() mutable
        {
          handler(e, T());
        });
  }
}

template <typename Executor, typename F, typename Handler>
awaiter<Executor>* co_spawn_entry_point(awaitable<void, Executor>*,
    executor_work_guard<Executor> work_guard, F f, Handler handler)
{
  std::exception_ptr e = nullptr;

  try
  {
    co_await f();
  }
  catch (...)
  {
    e = std::current_exception();
  }

  (dispatch)(work_guard.get_executor(),
      [handler = std::move(handler), e]() mutable
      {
        handler(e);
      });
}

template <typename Executor, typename F, typename CompletionToken>
auto co_spawn(const Executor& ex, F&& f, CompletionToken&& token)
{
  typedef typename result_of<F()>::type awaitable_type;
  typedef typename awaitable_type::executor_type executor_type;
  typedef typename awaitable_signature<awaitable_type>::type signature_type;

  async_completion<CompletionToken, signature_type> completion(token);

  executor_type ex2(ex);
  auto work_guard = make_work_guard(completion.completion_handler, ex2);

  auto* a = (co_spawn_entry_point)(
      static_cast<awaitable_type*>(nullptr), std::move(work_guard),
      std::forward<F>(f), std::move(completion.completion_handler));

  a->set_executor(ex2);
  (post)(co_spawn_handler<executor_type>(a));

  return completion.result.get();
}

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable:4033)
#endif // defined(_MSC_VER)

#if defined(_MSC_VER)
template <typename T> T dummy_return()
{
  return std::move(*static_cast<T*>(nullptr));
}

template <>
inline void dummy_return()
{
}
#endif // defined(_MSC_VER)

template <typename Awaitable>
inline Awaitable make_dummy_awaitable()
{
  for (;;) co_await std::experimental::suspend_always();
#if defined(_MSC_VER)
  co_return dummy_return<typename Awaitable::value_type>();
#endif // defined(_MSC_VER)
}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif // defined(_MSC_VER)

} // namespace detail
} // namespace experimental

template <typename Executor, typename R, typename... Args>
class async_result<experimental::await_token<Executor>, R(Args...)>
{
public:
  typedef experimental::detail::await_handler<
    Executor, typename decay<Args>::type...> completion_handler_type;

  typedef typename experimental::detail::await_handler<
    Executor, Args...>::awaitable_type return_type;

  async_result(completion_handler_type& h)
    : awaitable_(experimental::detail::make_dummy_awaitable<return_type>())
  {
    h.attach_awaitee(awaitable_);
  }

  return_type get()
  {
    return std::move(awaitable_);
  }

private:
  return_type awaitable_;
};

#if !defined(ASIO_NO_DEPRECATED)

template <typename Executor, typename R, typename... Args>
struct handler_type<experimental::await_token<Executor>, R(Args...)>
{
  typedef experimental::detail::await_handler<
    Executor, typename decay<Args>::type...> type;
};

template <typename Executor, typename... Args>
class async_result<experimental::detail::await_handler<Executor, Args...>>
{
public:
  typedef typename experimental::detail::await_handler<
    Executor, Args...>::awaitable_type type;

  async_result(experimental::detail::await_handler<Executor, Args...>& h)
    : awaitable_(experimental::detail::make_dummy_awaitable<type>())
  {
    h.attach_awaitee(awaitable_);
  }

  type get()
  {
    return std::move(awaitable_);
  }

private:
  type awaitable_;
};

#endif // !defined(ASIO_NO_DEPRECATED)

} // namespace asio

namespace std { namespace experimental {

template <typename Executor, typename... Args>
struct coroutine_traits<
  asio::experimental::detail::awaiter<Executor>*, Args...>
{
  typedef asio::experimental::detail::awaiter<Executor> promise_type;
};

template <typename T, typename Executor, typename... Args>
struct coroutine_traits<
  asio::experimental::awaitable<T, Executor>, Args...>
{
  typedef asio::experimental::detail::awaitee<T, Executor> promise_type;
};

}} // namespace std::experimental

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_IMPL_CO_SPAWN_HPP
