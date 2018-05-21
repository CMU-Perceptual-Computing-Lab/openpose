//
// impl/error_code.ipp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ERROR_CODE_IPP
#define ASIO_IMPL_ERROR_CODE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# include <winerror.h>
#elif defined(ASIO_WINDOWS_RUNTIME)
# include <windows.h>
#else
# include <cerrno>
# include <cstring>
# include <string>
#endif
#include "asio/detail/local_free_on_block_exit.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class system_category : public error_category
{
public:
  const char* name() const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    return "asio.system";
  }

  std::string message(int value) const
  {
#if defined(ASIO_WINDOWS_RUNTIME) || defined(ASIO_WINDOWS_APP)
    std::wstring wmsg(128, wchar_t());
    for (;;)
    {
      DWORD wlength = ::FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS, 0, value,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          &wmsg[0], static_cast<DWORD>(wmsg.size()), 0);
      if (wlength == 0 && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      {
        wmsg.resize(wmsg.size() + wmsg.size() / 2);
        continue;
      }
      if (wlength && wmsg[wlength - 1] == '\n')
        --wlength;
      if (wlength && wmsg[wlength - 1] == '\r')
        --wlength;
      if (wlength)
      {
        std::string msg(wlength * 2, char());
        int length = ::WideCharToMultiByte(CP_ACP, 0,
            wmsg.c_str(), static_cast<int>(wlength),
            &msg[0], static_cast<int>(wlength * 2), 0, 0);
        if (length <= 0)
          return "asio.system error";
        msg.resize(static_cast<std::size_t>(length));
        return msg;
      }
      else
        return "asio.system error";
    }
#elif defined(ASIO_WINDOWS) || defined(__CYGWIN__)
    char* msg = 0;
    DWORD length = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
        | FORMAT_MESSAGE_FROM_SYSTEM
        | FORMAT_MESSAGE_IGNORE_INSERTS, 0, value,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, 0);
    detail::local_free_on_block_exit local_free_obj(msg);
    if (length && msg[length - 1] == '\n')
      msg[--length] = '\0';
    if (length && msg[length - 1] == '\r')
      msg[--length] = '\0';
    if (length)
      return msg;
    else
      return "asio.system error";
#else // defined(ASIO_WINDOWS_DESKTOP) || defined(__CYGWIN__)
#if !defined(__sun)
    if (value == ECANCELED)
      return "Operation aborted.";
#endif // !defined(__sun)
#if defined(__sun) || defined(__QNX__) || defined(__SYMBIAN32__)
    using namespace std;
    return strerror(value);
#else
    char buf[256] = "";
    using namespace std;
    return strerror_result(strerror_r(value, buf, sizeof(buf)), buf);
#endif
#endif // defined(ASIO_WINDOWS_DESKTOP) || defined(__CYGWIN__)
  }

#if defined(ASIO_HAS_STD_ERROR_CODE)
  std::error_condition default_error_condition(
      int ev) const ASIO_ERROR_CATEGORY_NOEXCEPT
  {
    switch (ev)
    {
    case access_denied:
      return std::errc::permission_denied;
    case address_family_not_supported:
      return std::errc::address_family_not_supported;
    case address_in_use:
      return std::errc::address_in_use;
    case already_connected:
      return std::errc::already_connected;
    case already_started:
      return std::errc::connection_already_in_progress;
    case broken_pipe:
      return std::errc::broken_pipe;
    case connection_aborted:
      return std::errc::connection_aborted;
    case connection_refused:
      return std::errc::connection_refused;
    case connection_reset:
      return std::errc::connection_reset;
    case bad_descriptor:
      return std::errc::bad_file_descriptor;
    case fault:
      return std::errc::bad_address;
    case host_unreachable:
      return std::errc::host_unreachable;
    case in_progress:
      return std::errc::operation_in_progress;
    case interrupted:
      return std::errc::interrupted;
    case invalid_argument:
      return std::errc::invalid_argument;
    case message_size:
      return std::errc::message_size;
    case name_too_long:
      return std::errc::filename_too_long;
    case network_down:
      return std::errc::network_down;
    case network_reset:
      return std::errc::network_reset;
    case network_unreachable:
      return std::errc::network_unreachable;
    case no_descriptors:
      return std::errc::too_many_files_open;
    case no_buffer_space:
      return std::errc::no_buffer_space;
    case no_memory:
      return std::errc::not_enough_memory;
    case no_permission:
      return std::errc::operation_not_permitted;
    case no_protocol_option:
      return std::errc::no_protocol_option;
    case no_such_device:
      return std::errc::no_such_device;
    case not_connected:
      return std::errc::not_connected;
    case not_socket:
      return std::errc::not_a_socket;
    case operation_aborted:
      return std::errc::operation_canceled;
    case operation_not_supported:
      return std::errc::operation_not_supported;
    case shut_down:
      return std::make_error_condition(ev, *this);
    case timed_out:
      return std::errc::timed_out;
    case try_again:
      return std::errc::resource_unavailable_try_again;
    case would_block:
      return std::errc::operation_would_block;
    default:
      return std::make_error_condition(ev, *this);
  }
#endif // defined(ASIO_HAS_STD_ERROR_CODE)

private:
  // Helper function to adapt the result from glibc's variant of strerror_r.
  static const char* strerror_result(int, const char* s) { return s; }
  static const char* strerror_result(const char* s, const char*) { return s; }
};

} // namespace detail

const error_category& system_category()
{
  static detail::system_category instance;
  return instance;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_ERROR_CODE_IPP
