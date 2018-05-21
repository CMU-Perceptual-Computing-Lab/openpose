//
// error_code.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_CODE_HPP
#define ASIO_ERROR_CODE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <system_error>
#else // defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <string>
# include "asio/detail/noncopyable.hpp"
# if !defined(ASIO_NO_IOSTREAM)
#  include <iosfwd>
# endif // !defined(ASIO_NO_IOSTREAM)
#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::error_category error_category;

#else // defined(ASIO_HAS_STD_SYSTEM_ERROR)

/// Base class for all error categories.
class error_category : private noncopyable
{
public:
  /// Destructor.
  virtual ~error_category()
  {
  }

  /// Returns a string naming the error gategory.
  virtual const char* name() const = 0;

  /// Returns a string describing the error denoted by @c value.
  virtual std::string message(int value) const = 0;

  /// Equality operator to compare two error categories.
  bool operator==(const error_category& rhs) const
  {
    return this == &rhs;
  }

  /// Inequality operator to compare two error categories.
  bool operator!=(const error_category& rhs) const
  {
    return !(*this == rhs);
  }
};

#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

/// Returns the error category used for the system errors produced by asio.
extern ASIO_DECL const error_category& system_category();

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::error_code error_code;

#else // defined(ASIO_HAS_STD_SYSTEM_ERROR)

/// Class to represent an error code value.
class error_code
{
public:
  /// Default constructor.
  error_code()
    : value_(0),
      category_(&system_category())
  {
  }

  /// Construct with specific error code and category.
  error_code(int v, const error_category& c)
    : value_(v),
      category_(&c)
  {
  }

  /// Construct from an error code enum.
  template <typename ErrorEnum>
  error_code(ErrorEnum e)
  {
    *this = make_error_code(e);
  }

  /// Clear the error value to the default.
  void clear()
  {
    value_ = 0;
    category_ = &system_category();
  }

  /// Assign a new error value.
  void assign(int v, const error_category& c)
  {
    value_ = v;
    category_ = &c;
  }

  /// Get the error value.
  int value() const
  {
    return value_;
  }

  /// Get the error category.
  const error_category& category() const
  {
    return *category_;
  }

  /// Get the message associated with the error.
  std::string message() const
  {
    return category_->message(value_);
  }

  struct unspecified_bool_type_t
  {
  };

  typedef void (*unspecified_bool_type)(unspecified_bool_type_t);

  static void unspecified_bool_true(unspecified_bool_type_t) {}

  /// Operator returns non-null if there is a non-success error code.
  operator unspecified_bool_type() const
  {
    if (value_ == 0)
      return 0;
    else
      return &error_code::unspecified_bool_true;
  }

  /// Operator to test if the error represents success.
  bool operator!() const
  {
    return value_ == 0;
  }

  /// Equality operator to compare two error objects.
  friend bool operator==(const error_code& e1, const error_code& e2)
  {
    return e1.value_ == e2.value_ && e1.category_ == e2.category_;
  }

  /// Inequality operator to compare two error objects.
  friend bool operator!=(const error_code& e1, const error_code& e2)
  {
    return e1.value_ != e2.value_ || e1.category_ != e2.category_;
  }

private:
  // The value associated with the error code.
  int value_;

  // The category associated with the error code.
  const error_category* category_;
};

# if !defined(ASIO_NO_IOSTREAM)

/// Output an error code.
template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const error_code& ec)
{
  os << ec.category().name() << ':' << ec.value();
  return os;
}

# endif // !defined(ASIO_NO_IOSTREAM)

#endif // defined(ASIO_HAS_STD_SYSTEM_ERROR)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/error_code.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_ERROR_CODE_HPP
