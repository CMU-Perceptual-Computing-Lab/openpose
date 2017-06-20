#ifndef OPENPOSE_THREAD_ENUM_CLASSES_HPP
#define OPENPOSE_THREAD_ENUM_CLASSES_HPP

namespace op
{
    /**
     * ThreadManager synchronization mode.
     */
    enum class ThreadManagerMode : unsigned char
    {
        /**
         * First and last queues of ThreadManager will be given to the user, so he must push elements to the first queue and retrieve
         * them from the last one after being processed.
         * Recommended for prototyping environments (easier to test but more error-prone and potentially slower in performance).
         */
        Asynchronous,
        AsynchronousIn,     /**< Similar to Asynchronous, but only the input (first) queue is given to the user. */
        AsynchronousOut,    /**< Similar to Asynchronous, but only the output (last) queue is given to the user. */
        /**
         * Everything will run inside the ThreadManager.
         * Recommended for production environments (more difficult to set up but faster in performance and less error-prone).
         */
        Synchronous,
    };
}

#endif // OPENPOSE_THREAD_ENUM_CLASSES_HPP
