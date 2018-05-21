//
// detail/wrapped_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WRAPPED_HANDLER_HPP
#define ASIO_DETAIL_WRAPPED_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct is_continuation_delegated
{
  template <typename Dispatcher, typename Handler>
  bool operator()(Dispatcher&, Handler& handler) const
  {
    return asio_handler_cont_helpers::is_continuation(handler);
  }
};

struct is_continuation_if_running
{
  template <typename Dispatcher, typename Handler>
  bool operator()(Dispatcher& dispatcher, Handler&) const
  {
    return dispatcher.running_in_this_thread();
  }
};

template <typename Dispatcher, typename Handler,
    typename IsContinuation = is_continuation_delegated>
class wrapped_handler
{
public:
  typedef void result_type;

  wrapped_handler(Dispatcher dispatcher, Handler& handler)
    : dispatcher_(dispatcher),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
  }

#if defined(ASIO_HAS_MOVE)
  wrapped_handler(const wrapped_handler& other)
    : dispatcher_(other.dispatcher_),
      handler_(other.handler_)
  {
  }

  wrapped_handler(wrapped_handler&& other)
    : dispatcher_(other.dispatcher_),
      handler_(ASIO_MOVE_CAST(Handler)(other.handler_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  void operator()()
  {
    dispatcher_.dispatch(ASIO_MOVE_CAST(Handler)(handler_));
  }

  void operator()() const
  {
    dispatcher_.dispatch(handler_);
  }

  template <typename Arg1>
  void operator()(const Arg1& arg1)
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1));
  }

  template <typename Arg1>
  void operator()(const Arg1& arg1) const
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1));
  }

  template <typename Arg1, typename Arg2>
  void operator()(const Arg1& arg1, const Arg2& arg2)
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1, arg2));
  }

  template <typename Arg1, typename Arg2>
  void operator()(const Arg1& arg1, const Arg2& arg2) const
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1, arg2));
  }

  template <typename Arg1, typename Arg2, typename Arg3>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3)
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1, arg2, arg3));
  }

  template <typename Arg1, typename Arg2, typename Arg3>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3) const
  {
    dispatcher_.dispatch(detail::bind_handler(handler_, arg1, arg2, arg3));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
      const Arg4& arg4)
  {
    dispatcher_.dispatch(
        detail::bind_handler(handler_, arg1, arg2, arg3, arg4));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
      const Arg4& arg4) const
  {
    dispatcher_.dispatch(
        detail::bind_handler(handler_, arg1, arg2, arg3, arg4));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
      typename Arg5>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
      const Arg4& arg4, const Arg5& arg5)
  {
    dispatcher_.dispatch(
        detail::bind_handler(handler_, arg1, arg2, arg3, arg4, arg5));
  }

  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
      typename Arg5>
  void operator()(const Arg1& arg1, const Arg2& arg2, const Arg3& arg3,
      const Arg4& arg4, const Arg5& arg5) const
  {
    dispatcher_.dispatch(
        detail::bind_handler(handler_, arg1, arg2, arg3, arg4, arg5));
  }

//private:
  Dispatcher dispatcher_;
  Handler handler_;
};

template <typename Handler, typename Context>
class rewrapped_handler
{
public:
  explicit rewrapped_handler(Handler& handler, const Context& context)
    : context_(context),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
  }

  explicit rewrapped_handler(const Handler& handler, const Context& context)
    : context_(context),
      handler_(handler)
  {
  }

#if defined(ASIO_HAS_MOVE)
  rewrapped_handler(const rewrapped_handler& other)
    : context_(other.context_),
      handler_(other.handler_)
  {
  }

  rewrapped_handler(rewrapped_handler&& other)
    : context_(ASIO_MOVE_CAST(Context)(other.context_)),
      handler_(ASIO_MOVE_CAST(Handler)(other.handler_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  void operator()()
  {
    handler_();
  }

  void operator()() const
  {
    handler_();
  }

//private:
  Context context_;
  Handler handler_;
};

template <typename Dispatcher, typename Handler, typename IsContinuation>
inline void* asio_handler_allocate(std::size_t size,
    wrapped_handler<Dispatcher, Handler, IsContinuation>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Dispatcher, typename Handler, typename IsContinuation>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    wrapped_handler<Dispatcher, Handler, IsContinuation>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Dispatcher, typename Handler, typename IsContinuation>
inline bool asio_handler_is_continuation(
    wrapped_handler<Dispatcher, Handler, IsContinuation>* this_handler)
{
  return IsContinuation()(this_handler->dispatcher_, this_handler->handler_);
}

template <typename Function, typename Dispatcher,
    typename Handler, typename IsContinuation>
inline void asio_handler_invoke(Function& function,
    wrapped_handler<Dispatcher, Handler, IsContinuation>* this_handler)
{
  this_handler->dispatcher_.dispatch(
      rewrapped_handler<Function, Handler>(
        function, this_handler->handler_));
}

template <typename Function, typename Dispatcher,
    typename Handler, typename IsContinuation>
inline void asio_handler_invoke(const Function& function,
    wrapped_handler<Dispatcher, Handler, IsContinuation>* this_handler)
{
  this_handler->dispatcher_.dispatch(
      rewrapped_handler<Function, Handler>(
        function, this_handler->handler_));
}

template <typename Handler, typename Context>
inline void* asio_handler_allocate(std::size_t size,
    rewrapped_handler<Handler, Context>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->context_);
}

template <typename Handler, typename Context>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    rewrapped_handler<Handler, Context>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->context_);
}

template <typename Dispatcher, typename Context>
inline bool asio_handler_is_continuation(
    rewrapped_handler<Dispatcher, Context>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->context_);
}

template <typename Function, typename Handler, typename Context>
inline void asio_handler_invoke(Function& function,
    rewrapped_handler<Handler, Context>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->context_);
}

template <typename Function, typename Handler, typename Context>
inline void asio_handler_invoke(const Function& function,
    rewrapped_handler<Handler, Context>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->context_);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WRAPPED_HANDLER_HPP
