//
// detail/wince_thread.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINCE_THREAD_HPP
#define ASIO_DETAIL_WINCE_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS) && defined(UNDER_CE)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

DWORD WINAPI wince_thread_function(LPVOID arg);

class wince_thread
  : private noncopyable
{
public:
  // Constructor.
  template <typename Function>
  wince_thread(Function f, unsigned int = 0)
  {
    scoped_ptr<func_base> arg(new func<Function>(f));
    DWORD thread_id = 0;
    thread_ = ::CreateThread(0, 0, wince_thread_function,
        arg.get(), 0, &thread_id);
    if (!thread_)
    {
      DWORD last_error = ::GetLastError();
      asio::error_code ec(last_error,
          asio::error::get_system_category());
      asio::detail::throw_error(ec, "thread");
    }
    arg.release();
  }

  // Destructor.
  ~wince_thread()
  {
    ::CloseHandle(thread_);
  }

  // Wait for the thread to exit.
  void join()
  {
    ::WaitForSingleObject(thread_, INFINITE);
  }

  // Get number of CPUs.
  static std::size_t hardware_concurrency()
  {
    SYSTEM_INFO system_info;
    ::GetSystemInfo(&system_info);
    return system_info.dwNumberOfProcessors;
  }

private:
  friend DWORD WINAPI wince_thread_function(LPVOID arg);

  class func_base
  {
  public:
    virtual ~func_base() {}
    virtual void run() = 0;
  };

  template <typename Function>
  class func
    : public func_base
  {
  public:
    func(Function f)
      : f_(f)
    {
    }

    virtual void run()
    {
      f_();
    }

  private:
    Function f_;
  };

  ::HANDLE thread_;
};

inline DWORD WINAPI wince_thread_function(LPVOID arg)
{
  scoped_ptr<wince_thread::func_base> func(
      static_cast<wince_thread::func_base*>(arg));
  func->run();
  return 0;
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS) && defined(UNDER_CE)

#endif // ASIO_DETAIL_WINCE_THREAD_HPP
