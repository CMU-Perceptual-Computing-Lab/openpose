//
// detail/winrt_async_manager.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINRT_ASYNC_MANAGER_HPP
#define ASIO_DETAIL_WINRT_ASYNC_MANAGER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include <future>
#include "asio/detail/atomic_count.hpp"
#include "asio/detail/winrt_async_op.hpp"
#include "asio/error.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class winrt_async_manager
  : public asio::detail::service_base<winrt_async_manager>
{
public:
  // Constructor.
  winrt_async_manager(asio::io_context& io_context)
    : asio::detail::service_base<winrt_async_manager>(io_context),
      io_context_(use_service<io_context_impl>(io_context)),
      outstanding_ops_(1)
  {
  }

  // Destructor.
  ~winrt_async_manager()
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    if (--outstanding_ops_ > 0)
    {
      // Block until last operation is complete.
      std::future<void> f = promise_.get_future();
      f.wait();
    }
  }

  void sync(Windows::Foundation::IAsyncAction^ action,
      asio::error_code& ec)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto promise = std::make_shared<std::promise<asio::error_code>>();
    auto future = promise->get_future();

    action->Completed = ref new AsyncActionCompletedHandler(
      [promise](IAsyncAction^ action, AsyncStatus status)
      {
        switch (status)
        {
        case AsyncStatus::Canceled:
          promise->set_value(asio::error::operation_aborted);
          break;
        case AsyncStatus::Error:
        case AsyncStatus::Completed:
        default:
          asio::error_code ec(
              action->ErrorCode.Value,
              asio::system_category());
          promise->set_value(ec);
          break;
        }
      });

    ec = future.get();
  }

  template <typename TResult>
  TResult sync(Windows::Foundation::IAsyncOperation<TResult>^ operation,
      asio::error_code& ec)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto promise = std::make_shared<std::promise<asio::error_code>>();
    auto future = promise->get_future();

    operation->Completed = ref new AsyncOperationCompletedHandler<TResult>(
      [promise](IAsyncOperation<TResult>^ operation, AsyncStatus status)
      {
        switch (status)
        {
        case AsyncStatus::Canceled:
          promise->set_value(asio::error::operation_aborted);
          break;
        case AsyncStatus::Error:
        case AsyncStatus::Completed:
        default:
          asio::error_code ec(
              operation->ErrorCode.Value,
              asio::system_category());
          promise->set_value(ec);
          break;
        }
      });

    ec = future.get();
    return operation->GetResults();
  }

  template <typename TResult, typename TProgress>
  TResult sync(
      Windows::Foundation::IAsyncOperationWithProgress<
        TResult, TProgress>^ operation,
      asio::error_code& ec)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto promise = std::make_shared<std::promise<asio::error_code>>();
    auto future = promise->get_future();

    operation->Completed
      = ref new AsyncOperationWithProgressCompletedHandler<TResult, TProgress>(
        [promise](IAsyncOperationWithProgress<TResult, TProgress>^ operation,
          AsyncStatus status)
        {
          switch (status)
          {
          case AsyncStatus::Canceled:
            promise->set_value(asio::error::operation_aborted);
            break;
          case AsyncStatus::Started:
            break;
          case AsyncStatus::Error:
          case AsyncStatus::Completed:
          default:
            asio::error_code ec(
                operation->ErrorCode.Value,
                asio::system_category());
            promise->set_value(ec);
            break;
          }
        });

    ec = future.get();
    return operation->GetResults();
  }

  void async(Windows::Foundation::IAsyncAction^ action,
      winrt_async_op<void>* handler)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto on_completed = ref new AsyncActionCompletedHandler(
      [this, handler](IAsyncAction^ action, AsyncStatus status)
      {
        switch (status)
        {
        case AsyncStatus::Canceled:
          handler->ec_ = asio::error::operation_aborted;
          break;
        case AsyncStatus::Started:
          return;
        case AsyncStatus::Completed:
        case AsyncStatus::Error:
        default:
          handler->ec_ = asio::error_code(
              action->ErrorCode.Value,
              asio::system_category());
          break;
        }
        io_context_.post_deferred_completion(handler);
        if (--outstanding_ops_ == 0)
          promise_.set_value();
      });

    io_context_.work_started();
    ++outstanding_ops_;
    action->Completed = on_completed;
  }

  template <typename TResult>
  void async(Windows::Foundation::IAsyncOperation<TResult>^ operation,
      winrt_async_op<TResult>* handler)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto on_completed = ref new AsyncOperationCompletedHandler<TResult>(
      [this, handler](IAsyncOperation<TResult>^ operation, AsyncStatus status)
      {
        switch (status)
        {
        case AsyncStatus::Canceled:
          handler->ec_ = asio::error::operation_aborted;
          break;
        case AsyncStatus::Started:
          return;
        case AsyncStatus::Completed:
          handler->result_ = operation->GetResults();
          // Fall through.
        case AsyncStatus::Error:
        default:
          handler->ec_ = asio::error_code(
              operation->ErrorCode.Value,
              asio::system_category());
          break;
        }
        io_context_.post_deferred_completion(handler);
        if (--outstanding_ops_ == 0)
          promise_.set_value();
      });

    io_context_.work_started();
    ++outstanding_ops_;
    operation->Completed = on_completed;
  }

  template <typename TResult, typename TProgress>
  void async(
      Windows::Foundation::IAsyncOperationWithProgress<
        TResult, TProgress>^ operation,
      winrt_async_op<TResult>* handler)
  {
    using namespace Windows::Foundation;
    using Windows::Foundation::AsyncStatus;

    auto on_completed
      = ref new AsyncOperationWithProgressCompletedHandler<TResult, TProgress>(
        [this, handler](IAsyncOperationWithProgress<
          TResult, TProgress>^ operation, AsyncStatus status)
        {
          switch (status)
          {
          case AsyncStatus::Canceled:
            handler->ec_ = asio::error::operation_aborted;
            break;
          case AsyncStatus::Started:
            return;
          case AsyncStatus::Completed:
            handler->result_ = operation->GetResults();
            // Fall through.
          case AsyncStatus::Error:
          default:
            handler->ec_ = asio::error_code(
                operation->ErrorCode.Value,
                asio::system_category());
            break;
          }
          io_context_.post_deferred_completion(handler);
          if (--outstanding_ops_ == 0)
            promise_.set_value();
        });

    io_context_.work_started();
    ++outstanding_ops_;
    operation->Completed = on_completed;
  }

private:
  // The io_context implementation used to post completed handlers.
  io_context_impl& io_context_;

  // Count of outstanding operations.
  atomic_count outstanding_ops_;

  // Used to keep wait for outstanding operations to complete.
  std::promise<void> promise_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_ASYNC_MANAGER_HPP
