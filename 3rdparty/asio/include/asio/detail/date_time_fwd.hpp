//
// detail/date_time_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_DATE_TIME_FWD_HPP
#define ASIO_DETAIL_DATE_TIME_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

namespace boost {
namespace date_time {

template<class T, class TimeSystem>
class base_time;

} // namespace date_time
namespace posix_time {

class ptime;

} // namespace posix_time
} // namespace boost

#endif // ASIO_DETAIL_DATE_TIME_FWD_HPP
