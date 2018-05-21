//
// deadline_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEADLINE_TIMER_SERVICE_HPP
#define ASIO_DEADLINE_TIMER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_OLD_SERVICES)

#if defined(ASIO_HAS_BOOST_DATE_TIME) \
  || defined(GENERATING_DOCUMENTATION)

#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/detail/deadline_timer_service.hpp"
#include "asio/io_context.hpp"
#include "asio/time_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default service implementation for a timer.
template <typename TimeType,
    typename TimeTraits = asio::time_traits<TimeType> >
class deadline_timer_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_context::service
#else
  : public asio::detail::service_base<
      deadline_timer_service<TimeType, TimeTraits> >
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_context::id id;
#endif

  /// The time traits type.
  typedef TimeTraits traits_type;

  /// The time type.
  typedef typename traits_type::time_type time_type;

  /// The duration type.
  typedef typename traits_type::duration_type duration_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::deadline_timer_service<traits_type> service_impl_type;

public:
  /// The implementation type of the deadline timer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// Construct a new timer service for the specified io_context.
  explicit deadline_timer_service(asio::io_context& io_context)
    : asio::detail::service_base<
        deadline_timer_service<TimeType, TimeTraits> >(io_context),
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

  /// Get the expiry time for the timer as an absolute time.
  time_type expires_at(const implementation_type& impl) const
  {
    return service_impl_.expiry(impl);
  }

  /// Set the expiry time for the timer as an absolute time.
  std::size_t expires_at(implementation_type& impl,
      const time_type& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_at(impl, expiry_time, ec);
  }

  /// Get the expiry time for the timer relative to now.
  duration_type expires_from_now(const implementation_type& impl) const
  {
    return TimeTraits::subtract(service_impl_.expiry(impl), TimeTraits::now());
  }

  /// Set the expiry time for the timer relative to now.
  std::size_t expires_from_now(implementation_type& impl,
      const duration_type& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_after(impl, expiry_time, ec);
  }

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

#endif // defined(ASIO_HAS_BOOST_DATE_TIME)
       // || defined(GENERATING_DOCUMENTATION)

#endif // defined(ASIO_ENABLE_OLD_SERVICES)

#endif // ASIO_DEADLINE_TIMER_SERVICE_HPP
