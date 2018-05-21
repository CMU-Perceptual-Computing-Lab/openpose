//
// detail/handler_tracking.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_TRACKING_HPP
#define ASIO_DETAIL_HANDLER_TRACKING_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

namespace asio {

class execution_context;

} // namespace asio

#if defined(ASIO_CUSTOM_HANDLER_TRACKING)
# include ASIO_CUSTOM_HANDLER_TRACKING
#elif defined(ASIO_ENABLE_HANDLER_TRACKING)
# include "asio/error_code.hpp"
# include "asio/detail/cstdint.hpp"
# include "asio/detail/static_mutex.hpp"
# include "asio/detail/tss_ptr.hpp"
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_CUSTOM_HANDLER_TRACKING)

// The user-specified header must define the following macros:
// - ASIO_INHERIT_TRACKED_HANDLER
// - ASIO_ALSO_INHERIT_TRACKED_HANDLER
// - ASIO_HANDLER_TRACKING_INIT
// - ASIO_HANDLER_CREATION(args)
// - ASIO_HANDLER_COMPLETION(args)
// - ASIO_HANDLER_INVOCATION_BEGIN(args)
// - ASIO_HANDLER_INVOCATION_END
// - ASIO_HANDLER_OPERATION(args)
// - ASIO_HANDLER_REACTOR_REGISTRATION(args)
// - ASIO_HANDLER_REACTOR_DEREGISTRATION(args)
// - ASIO_HANDLER_REACTOR_READ_EVENT
// - ASIO_HANDLER_REACTOR_WRITE_EVENT
// - ASIO_HANDLER_REACTOR_ERROR_EVENT
// - ASIO_HANDLER_REACTOR_EVENTS(args)
// - ASIO_HANDLER_REACTOR_OPERATION(args)

# if !defined(ASIO_ENABLE_HANDLER_TRACKING)
#  define ASIO_ENABLE_HANDLER_TRACKING 1
# endif /// !defined(ASIO_ENABLE_HANDLER_TRACKING)

#elif defined(ASIO_ENABLE_HANDLER_TRACKING)

class handler_tracking
{
public:
  class completion;

  // Base class for objects containing tracked handlers.
  class tracked_handler
  {
  private:
    // Only the handler_tracking class will have access to the id.
    friend class handler_tracking;
    friend class completion;
    uint64_t id_;

  protected:
    // Constructor initialises with no id.
    tracked_handler() : id_(0) {}

    // Prevent deletion through this type.
    ~tracked_handler() {}
  };

  // Initialise the tracking system.
  ASIO_DECL static void init();

  // Record the creation of a tracked handler.
  ASIO_DECL static void creation(
      execution_context& context, tracked_handler& h,
      const char* object_type, void* object,
      uintmax_t native_handle, const char* op_name);

  class completion
  {
  public:
    // Constructor records that handler is to be invoked with no arguments.
    ASIO_DECL explicit completion(const tracked_handler& h);

    // Destructor records only when an exception is thrown from the handler, or
    // if the memory is being freed without the handler having been invoked.
    ASIO_DECL ~completion();

    // Records that handler is to be invoked with no arguments.
    ASIO_DECL void invocation_begin();

    // Records that handler is to be invoked with one arguments.
    ASIO_DECL void invocation_begin(const asio::error_code& ec);

    // Constructor records that handler is to be invoked with two arguments.
    ASIO_DECL void invocation_begin(
        const asio::error_code& ec, std::size_t bytes_transferred);

    // Constructor records that handler is to be invoked with two arguments.
    ASIO_DECL void invocation_begin(
        const asio::error_code& ec, int signal_number);

    // Constructor records that handler is to be invoked with two arguments.
    ASIO_DECL void invocation_begin(
        const asio::error_code& ec, const char* arg);

    // Record that handler invocation has ended.
    ASIO_DECL void invocation_end();

  private:
    friend class handler_tracking;
    uint64_t id_;
    bool invoked_;
    completion* next_;
  };

  // Record an operation that is not directly associated with a handler.
  ASIO_DECL static void operation(execution_context& context,
      const char* object_type, void* object,
      uintmax_t native_handle, const char* op_name);

  // Record that a descriptor has been registered with the reactor.
  ASIO_DECL static void reactor_registration(execution_context& context,
      uintmax_t native_handle, uintmax_t registration);

  // Record that a descriptor has been deregistered from the reactor.
  ASIO_DECL static void reactor_deregistration(execution_context& context,
      uintmax_t native_handle, uintmax_t registration);

  // Record a reactor-based operation that is associated with a handler.
  ASIO_DECL static void reactor_events(execution_context& context,
      uintmax_t registration, unsigned events);

  // Record a reactor-based operation that is associated with a handler.
  ASIO_DECL static void reactor_operation(
      const tracked_handler& h, const char* op_name,
      const asio::error_code& ec);

  // Record a reactor-based operation that is associated with a handler.
  ASIO_DECL static void reactor_operation(
      const tracked_handler& h, const char* op_name,
      const asio::error_code& ec, std::size_t bytes_transferred);

  // Write a line of output.
  ASIO_DECL static void write_line(const char* format, ...);

private:
  struct tracking_state;
  ASIO_DECL static tracking_state* get_state();
};

# define ASIO_INHERIT_TRACKED_HANDLER \
  : public asio::detail::handler_tracking::tracked_handler

# define ASIO_ALSO_INHERIT_TRACKED_HANDLER \
  , public asio::detail::handler_tracking::tracked_handler

# define ASIO_HANDLER_TRACKING_INIT \
  asio::detail::handler_tracking::init()

# define ASIO_HANDLER_CREATION(args) \
  asio::detail::handler_tracking::creation args

# define ASIO_HANDLER_COMPLETION(args) \
  asio::detail::handler_tracking::completion tracked_completion args

# define ASIO_HANDLER_INVOCATION_BEGIN(args) \
  tracked_completion.invocation_begin args

# define ASIO_HANDLER_INVOCATION_END \
  tracked_completion.invocation_end()

# define ASIO_HANDLER_OPERATION(args) \
  asio::detail::handler_tracking::operation args

# define ASIO_HANDLER_REACTOR_REGISTRATION(args) \
  asio::detail::handler_tracking::reactor_registration args

# define ASIO_HANDLER_REACTOR_DEREGISTRATION(args) \
  asio::detail::handler_tracking::reactor_deregistration args

# define ASIO_HANDLER_REACTOR_READ_EVENT 1
# define ASIO_HANDLER_REACTOR_WRITE_EVENT 2
# define ASIO_HANDLER_REACTOR_ERROR_EVENT 4

# define ASIO_HANDLER_REACTOR_EVENTS(args) \
  asio::detail::handler_tracking::reactor_events args

# define ASIO_HANDLER_REACTOR_OPERATION(args) \
  asio::detail::handler_tracking::reactor_operation args

#else // defined(ASIO_ENABLE_HANDLER_TRACKING)

# define ASIO_INHERIT_TRACKED_HANDLER
# define ASIO_ALSO_INHERIT_TRACKED_HANDLER
# define ASIO_HANDLER_TRACKING_INIT (void)0
# define ASIO_HANDLER_CREATION(args) (void)0
# define ASIO_HANDLER_COMPLETION(args) (void)0
# define ASIO_HANDLER_INVOCATION_BEGIN(args) (void)0
# define ASIO_HANDLER_INVOCATION_END (void)0
# define ASIO_HANDLER_OPERATION(args) (void)0
# define ASIO_HANDLER_REACTOR_REGISTRATION(args) (void)0
# define ASIO_HANDLER_REACTOR_DEREGISTRATION(args) (void)0
# define ASIO_HANDLER_REACTOR_READ_EVENT 0
# define ASIO_HANDLER_REACTOR_WRITE_EVENT 0
# define ASIO_HANDLER_REACTOR_ERROR_EVENT 0
# define ASIO_HANDLER_REACTOR_EVENTS(args) (void)0
# define ASIO_HANDLER_REACTOR_OPERATION(args) (void)0

#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/handler_tracking.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_HANDLER_TRACKING_HPP
