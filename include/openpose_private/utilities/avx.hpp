#ifndef OPENPOSE_PRIVATE_UTILITIES_AVX_HPP
#define OPENPOSE_PRIVATE_UTILITIES_AVX_HPP

// Warning:
// This file contains auxiliary functions for AVX.
// This file should only be included from cpp files.
// Default #include <openpose/headers.hpp> does not include it.

#ifdef WITH_AVX
    #include <cstdint> // uintptr_t
    #include <memory> // shared_ptr
    #include <immintrin.h>
    #include <openpose/utilities/errorAndLog.hpp>

    namespace op
    {
        #ifdef __GNUC__
            #define ALIGN32(x) x __attribute__((aligned(32)))
        #elif defined(_MSC_VER) // defined(_WIN32)
            #define ALIGN32(x) __declspec(align(32))
        #else
            #error Unknown environment!
        #endif

        // Functions
        // Sources:
        // - https://stackoverflow.com/questions/32612190/how-to-solve-the-32-byte-alignment-issue-for-avx-load-store-operations
        // - https://embeddedartistry.com/blog/2017/2/20/implementing-aligned-malloc
        // - https://embeddedartistry.com/blog/2017/2/23/c-smart-pointers-with-aligned-mallocfree
        typedef unsigned long long offset_t;
        #define PTR_OFFSET_SZ sizeof(offset_t)
        #ifndef align_up
        #define align_up(num, align) \
            (((num) + ((align) - 1)) & ~((align) - 1))
        #endif
        inline void * aligned_malloc(const size_t align, const size_t size)
        {
            void * ptr = nullptr;

            // 2 conditions:
            //  - We want both align and size to be greater than 0
            //  - We want it to be a power of two since align_up operates on powers of two
            if (align && size && (align & (align - 1)) == 0)
            {
                // We know we have to fit an offset value
                // We also allocate extra bytes to ensure we can meet the alignment
                const auto hdr_size = PTR_OFFSET_SZ + (align - 1);
                void * p = malloc(size + hdr_size);

                if (p)
                {
                    // Add the offset size to malloc's pointer (we will always store that)
                    // Then align the resulting value to the arget alignment
                    ptr = (void *) align_up(((uintptr_t)p + PTR_OFFSET_SZ), align);

                    // Calculate the offset and store it behind our aligned pointer
                    *((offset_t *)ptr - 1) = (offset_t)((uintptr_t)ptr - (uintptr_t)p);

                } // else nullptr, could not malloc
            } // else nullptr, invalid arguments

            if (ptr == nullptr)
            {
                error("Shared pointer could not be allocated for Array data storage.",
                      __LINE__, __FUNCTION__, __FILE__);
            }

            return ptr;
        }
        inline void aligned_free(void * ptr)
        {
            if (ptr == nullptr)
                error("Received nullptr.", __LINE__, __FUNCTION__, __FILE__);

            // Walk backwards from the passed-in pointer to get the pointer offset
            // We convert to an offset_t pointer and rely on pointer math to get the data
            offset_t offset = *((offset_t *)ptr - 1);

            // Once we have the offset, we can get our original pointer and call free
            void * p = (void *)((uint8_t *)ptr - offset);
            free(p);
        }
        template<class T>
        std::shared_ptr<T> aligned_shared_ptr(const size_t size)
        {
            try
            {
                return std::shared_ptr<T>(static_cast<T*>(
                    aligned_malloc(8*sizeof(T), sizeof(T)*size)), &aligned_free);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return std::shared_ptr<T>{};
            }
        }
    }
#endif

#endif // OPENPOSE_PRIVATE_UTILITIES_AVX_HPP
