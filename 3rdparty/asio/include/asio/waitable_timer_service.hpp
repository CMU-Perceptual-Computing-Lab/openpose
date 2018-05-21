//
// waitable_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WAITABLE_TIMER_SERVICE_HPP
#define ASIO_WAITABLE_TIMER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/detail/chrono_time_traits.hpp"
#include "asio/detail/deadline_timer_service.hpp"
#include "asio/io_context.hpp"
#include "asio/wait_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default service implementation for a timer.
template <typename Clock,
    typename WaitTraits = asio::wait_traits<Clock> >
class waitable_timer_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<
      waitable_timer_service<Clock, WaitTraits> >
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

  /// The clock type.
  typedef Clock clock_type;

  /// The duration type of the clock.
  typedef typename clock_type::duration duration;

  /// The time point type of the clock.
  typedef typename clock_type::time_point time_point;

  /// The wait traits type.
  typedef WaitTraits traits_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::deadline_timer_service<
    detail::chrono_time_traits<Clock, WaitTraits> > service_impl_type;

public:
  /// The implementation type of the waitable timer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// Construct a new timer service for the specified io_context.
  explicit waitable_timer_service(asio::io_context& io_context)
    : asio::detail::service_base<
        waitable_timer_service<Clock, WaitTraits> >(io_context),
      service_impl_(io_context)
  {
  }

  /// Construct a new timer implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a timer implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a new timer implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl)
  {
    service_impl_.move_construct(impl, other_impl);
  }

  /// Move-assign from another timer implementation.
  void move_assign(implementation_type& impl,
      waitable_timer_service& other_service,
      implementation_type& other_impl)
  {
    service_impl_.move_assign(impl, other_service.service_impl_, other_impl);
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Cancel any asynchronous wait operations associated with the timer.
  std::size_t cancel(implementation_type& impl, asio::error_code& ec)
  {
    return service_impl_.cancel(impl, ec);
  }

  /// Cancels one asynchronous wait operation associated with the timer.
  std::size_t cancel_one(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.cancel_one(impl, ec);
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use expiry().) Get the expiry time for the timer as an
  /// absolute time.
  time_point expires_at(const implementation_type& impl) const
  {
    return service_impl_.expiry(impl);
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Get the expiry time for the timer as an absolute time.
  time_point expiry(const implementation_type& impl) const
  {
    return service_impl_.expiry(impl);
  }

  /// Set the expiry time for the timer as an absolute time.
  std::size_t expires_at(implementation_type& impl,
      const time_point& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_at(impl, expiry_time, ec);
  }

  /// Set the expiry time for the timer relative to now.
  std::size_t expires_after(implementation_type& impl,
      const duration& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_after(impl, expiry_time, ec);
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated: Use expiry().) Get the expiry time for the timer relative to
  /// now.
  duration expires_from_now(const implementation_type& impl) const
  {
    typedef detail::chrono_time_traits<Clock, WaitTraits> traits;
    return traits::subtract(service_impl_.expiry(impl), traits::now());
  }

  /// (Deprecated: Use expires_after().) Set the expiry time for the timer
  /// relative to now.
  std::size_t expires_from_now(implementation_type& impl,
      const duration& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_after(impl, expiry_time, ec);
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  // Perform a blocking wait on the timer.
  void wait(implementation_type& impl, asio::error_code& ec)
  {
    service_impl_.wait(impl, ec);
  }

  // Start an asynchronous wait on the timer.
  template <typename WaitHandler>
  ASIO_INITFN_RESULT_TYPE(WaitHandler,
      void (asio::error_code))
  async_wait(implementation_type& impl,
      ASIO_MOVE_ARG(WaitHandler) handler)
  {
    async_completion<WaitHandler,
      void (asio::error_code)> init(handler);

    service_impl_.async_wait(impl, init.completion_handler);

    return init.result.get();
  }

private:
  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    service_impl_.shutdown();
  }

  // The platform-specific implementation.
  service_impl_type service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_WAITABLE_TIMER_SERVICE_HPP
