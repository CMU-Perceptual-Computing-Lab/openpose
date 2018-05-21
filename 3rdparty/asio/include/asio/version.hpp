//
// version.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_VERSION_HPP
#define ASIO_VERSION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

// ASIO_VERSION % 100 is the sub-minor version
// ASIO_VERSION / 100 % 1000 is the minor version
// ASIO_VERSION / 100000 is the major version
#define ASIO_VERSION 101201 // 1.12.1

#endif // ASIO_VERSION_HPP
