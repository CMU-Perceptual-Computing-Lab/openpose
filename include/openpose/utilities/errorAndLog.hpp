#ifndef OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
#define OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP

#include <sstream> // std::stringstream
#include <string>
#include <vector>
#include <openpose/core/macros.hpp>
#include <openpose/utilities/enumClasses.hpp>

namespace op
{
    OP_API void setMainThread();

    OP_API std::string getThreadId();

    OP_API bool getIfInMainThreadOrEmpty();

    OP_API bool getIfNotInMainThreadOrEmpty();

    template<typename T>
    std::string tToString(const T& message)
    {
        // Message -> ostringstream
        std::ostringstream oss;
        oss << message;
        // ostringstream -> std::string
        return oss.str();
    }

    /**
     * Differences between different kind of errrors:
     *  - error() is a normal error in the code.
     *  - errorWorker() is an error that occurred on a thread. Therefore, the machine will stop the threads, go back
     *    to the main thread, and then throw the error.
     *  - errorDestructor() is an error that occurred on a destructor. Exception on destructors provokes core dumped,
     *    so we simply output an error message via std::cerr.
     */

    // Error managment - How to use:
        // error(message, __LINE__, __FUNCTION__, __FILE__);
    OP_API void error(
        const std::string& message, const int line = -1, const std::string& function = "",
        const std::string& file = "");

    template<typename T>
    inline void error(
        const T& message, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        error(tToString(message), line, function, file);
    }

    // Worker error managment
    OP_API void checkWorkerErrors();

    OP_API void errorWorker(
        const std::string& message, const int line = -1, const std::string& function = "",
        const std::string& file = "");

    template<typename T>
    inline void errorWorker(
        const T& message, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        errorWorker(tToString(message), line, function, file);
    }

    // Destructor error managment
    OP_API void errorDestructor(
        const std::string& message, const int line = -1, const std::string& function = "",
        const std::string& file = "");

    template<typename T>
    inline void errorDestructor(
        const T& message, const int line = -1, const std::string& function = "", const std::string& file = "")
    {
        errorDestructor(tToString(message), line, function, file);
    }

    // Printing info - How to use:
        // It will print info if desiredPriority >= sPriorityThreshold
        // log(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);
    OP_API void log(
        const std::string& message, const Priority priority = Priority::Max, const int line = -1,
        const std::string& function = "", const std::string& file = "");

    template<typename T>
    inline void log(
        const T& message, const Priority priority = Priority::Max, const int line = -1,
        const std::string& function = "", const std::string& file = "")
    {
        log(tToString(message), priority, line, function, file);
    }

    // If only desired on debug mode (no computational cost at all on release mode):
        // It will print info if desiredPriority >= sPriorityThreshold
        // dLog(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);
    template<typename T>
    inline void dLog(
        const T& message, const Priority priority = Priority::Max, const int line = -1,
        const std::string& function = "", const std::string& file = "")
    {
        #ifndef NDEBUG
            log(message, priority, line, function, file);
        #else
            UNUSED(message);
            UNUSED(priority);
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
        #endif
    }

    // This class is thread-safe
    namespace ConfigureError
    {
        OP_API std::vector<ErrorMode> getErrorModes();

        OP_API void setErrorModes(const std::vector<ErrorMode>& errorModes);
    }

    // This class is not fully thread-safe
    namespace ConfigureLog
    {
        OP_API Priority getPriorityThreshold();

        OP_API const std::vector<LogMode>& getLogModes();

        // This function is not thread-safe. It must be run at the beginning
        OP_API void setPriorityThreshold(const Priority priorityThreshold);

        // This function is not thread-safe. It must be run at the beginning
        OP_API void setLogModes(const std::vector<LogMode>& loggingModes);
    }
}

#endif // OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
