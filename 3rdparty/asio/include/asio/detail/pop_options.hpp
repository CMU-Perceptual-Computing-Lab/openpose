//
// detail/pop_options.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// No header guard

#if defined(__COMO__)

// Comeau C++

#elif defined(__DMC__)

// Digital Mars C++

#elif defined(__INTEL_COMPILER) || defined(__ICL) \
  || defined(__ICC) || defined(__ECC)

// Intel C++

# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  pragma GCC visibility pop
# endif // (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)

#elif defined(__clang__)

// Clang

# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if defined(ASIO_OBJC_WORKAROUND)
#    undef Protocol
#    undef id
#    undef ASIO_OBJC_WORKAROUND
#   endif
#  endif
# endif

# if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
#  pragma GCC visibility pop
# endif // !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)

#elif defined(__GNUC__)

// GNU C++

# if defined(__MINGW32__) || defined(__CYGWIN__)
#  pragma pack (pop)
# endif

# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if defined(ASIO_OBJC_WORKAROUND)
#    undef Protocol
#    undef id
#    undef ASIO_OBJC_WORKAROUND
#   endif
#  endif
# endif

# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  pragma GCC visibility pop
# endif // (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)

# if (__GNUC__ >= 7)
#  pragma GCC diagnostic pop
# endif // (__GNUC__ >= 7)

#elif defined(__KCC)

// Kai C++

#elif defined(__sgi)

// SGI MIPSpro C++

#elif defined(__DECCXX)

// Compaq Tru64 Unix cxx

#elif defined(__ghs)

// Greenhills C++

#elif defined(__BORLANDC__)

// Borland C++

# pragma option pop
# pragma nopushoptwarn
# pragma nopackwarning

#elif defined(__MWERKS__)

// Metrowerks CodeWarrior

#elif defined(__SUNPRO_CC)

// Sun Workshop Compiler C++

#elif defined(__HP_aCC)

// HP aCC

#elif defined(__MRC__) || defined(__SC__)

// MPW MrCpp or SCpp

#elif defined(__IBMCPP__)

// IBM Visual Age

#elif defined(_MSC_VER)

// Microsoft Visual C++
//
// Must remain the last #elif since some other vendors (Metrowerks, for example)
// also #define _MSC_VER

# pragma warning (pop)
# pragma pack (pop)

# if defined(__cplusplus_cli) || defined(__cplusplus_winrt)
#  if defined(ASIO_CLR_WORKAROUND)
#   undef generic
#   undef ASIO_CLR_WORKAROUND
#  endif
# endif

#endif
