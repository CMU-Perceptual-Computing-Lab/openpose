#ifndef OPENPOSE_CORE_COMMON_HPP
#define OPENPOSE_CORE_COMMON_HPP

// Std library most used classes
#include <array>
#include <memory> // std::shared_ptr, std::unique_ptr
#include <string>
#include <vector>
// OpenPose most used classes
#include <openpose/core/array.hpp>
#include <openpose/core/macros.hpp>
#include <openpose/core/point.hpp>
#include <openpose/core/rectangle.hpp>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/profiler.hpp>
// Datum at the end, otherwise circular dependency with array, point & rectangle
#include <openpose/core/datum.hpp>

#endif // OPENPOSE_CORE_COMMON_HPP
