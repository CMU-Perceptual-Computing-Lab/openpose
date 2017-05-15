#ifndef CAFFE_EXPORT_HPP_
#define CAFFE_EXPORT_HPP_

// CAFFE_BUILDING_STATIC_LIB should be defined 
// only by the caffe target
#if defined(_MSC_VER) && !defined(CAFFE_BUILDING_STATIC_LIB) 
    #include "caffe/include_symbols.hpp"
#endif

#endif  // CAFFE_EXPORT_HPP_
