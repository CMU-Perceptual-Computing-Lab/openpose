//
// system_error.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SYSTEM_ERROR_HPP
#define ASIO_SYSTEM_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <system_error>
#else // defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <cerrno>
# include <exception>
# include <string>
# include "asio/error_code.hpp"
# include "asio/detail/scoped_ptr.hpp"
#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::system_error system_error;

#else // defined(ASIO_HAS_STD_SYSTEM_ERROR)

/// The system_error class is used to represent system conditions that
/// prevent the library from operating correctly.
class system_error
  : public std::exception
{
public:
  /// Construct with an error code.
  system_error(const error_code& ec)
    : code_(ec),
      context_()
  {
  }

  /// Construct with an error code and context.
  system_error(const error_code& ec, const std::string& context)
    : code_(ec),
      context_(context)
  {
  }

  /// Copy constructor.
  system_error(const system_error& other)
    : std::exception(other),
      code_(other.code_),
      context_(other.context_),
      what_()
  {
  }

  /// Destructor.
  virtual ~system_error() throw ()
  {
  }

  /// Assignment operator.
  system_error& operator=(const system_error& e)
  {
    context_ = e.context_;
    code_ = e.code_;
    what_.reset();
    return *this;
  }

  /// Get a string representation of the exception.
  virtual const char* what() const throw ()
  {
#if !defined(ASIO_NO_EXCEPTIONS)
    try
#endif // !defined(ASIO_NO_EXCEPTIONS)
    {
      if (!what_.get())
      {
        std::string tmp(context_);
        if (tmp.length())
          tmp += ": ";
        tmp += code_.message();
        what_.reset(new std::string(tmp));
      }
      return what_->c_str();
    }
#if !defined(ASIO_NO_EXCEPTIONS)
    catch (std::exception&)
    {
      return "system_error";
    }
#endif // !defined(ASIO_NO_EXCEPTIONS)
  }

  /// Get the error code associated with the exception.
  error_code code() const
  {
    return code_;
  }

private:
  // The code associated with the error.
  error_code code_;

  // The context associated with the error.
  std::string context_;

  // The string representation of the error.
  mutable asio::detail::scoped_ptr<std::string> what_;
};

#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SYSTEM_ERROR_HPP
