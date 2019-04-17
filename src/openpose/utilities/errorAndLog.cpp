#include <atomic>
#include <mutex>
#include <ctime> // std::tm, std::time_t
#include <fstream> // std::ifstream, std::ofstream
#include <iostream> // std::cout, std::endl
#include <stdexcept> // std::runtime_error
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
    std::atomic<bool> sThreadErrors;
    std::mutex sMutex;
    std::vector<std::string> sThreadErrorMessages;
    std::string sMainThreadId;

    #ifdef USE_UNITY_SUPPORT
        namespace UnityDebugger
        {
            #ifdef _WIN32
                typedef void(__stdcall * DebugCallback) (const char* const str, int type);
                DebugCallback unityDebugCallback;
            #endif
            bool unityDebugEnabled = true;

            #ifdef _WIN32
                extern "C" void OP_API _OPRegisterDebugCallback(DebugCallback debugCallback)
                {
                    if (debugCallback)
                    unityDebugCallback = debugCallback;
                }

                extern "C" void OP_API _OPSetDebugEnable(bool enable)
                {
                    unityDebugEnabled = enable;
                }
            #endif

            void DebugInUnity(const std::string& message, const int type)
            {
                #ifdef _WIN32
                    if (unityDebugEnabled)
                        if (unityDebugCallback)
                            unityDebugCallback(message.c_str(), type);
                #else
                    UNUSED(message);
                    UNUSED(type);
                    error("Unity plugin only available on Windows.", __LINE__, __FUNCTION__, __FILE__);
                #endif
            }

            void log(const std::string& message) { DebugInUnity(message, 0); }
            void logWarning(const std::string& message) { DebugInUnity(message, 1); }
            void logError(const std::string& message) { DebugInUnity(message, -1); }
        }
    #endif

    // Private auxiliar functions
    bool checkIfErrorHas(const ErrorMode errorMode)
    {
        for (const auto& sErrorMode : ConfigureError::getErrorModes())
            if (sErrorMode == errorMode || sErrorMode == ErrorMode::All)
                return true;
        return false;
    }

    bool checkIfLoggingHas(const LogMode loggingMode)
    {
        for (const auto& sLoggingMode : ConfigureLog::getLogModes())
            if (sLoggingMode == loggingMode || sLoggingMode == LogMode::All)
                return true;
        return false;
    }

    // Note: createFullMessage(message) = message
    std::string createFullMessage(const std::string& message, const int line = -1, const std::string& function = "",
                                  const std::string& file = "")
    {
        const auto hasMessage = (!message.empty());
        const auto hasLocation = (line != -1 || !function.empty() || !file.empty());

        std::string fullMessage;

        if (hasMessage)
        {
            fullMessage += message;
            // // Add dot at the end if the sentence does not finish in a programming file path (this happens when the
            // error is propagated over several error)
            // if (*message.crbegin() != '.'
            //     && (message.size() < 4
            //         || (message.substr(message.size() - 4, 4) != ".cpp"
            //             && message.substr(message.size() - 4, 4) != ".hpp"
            //             && message.substr(message.size() - 4, 4) != ".h"
            //             && message.substr(message.size() - 4, 4) != ".c")))
            //     fullMessage += ".";

            if (hasLocation)
            {
                if (*message.crbegin() != '.')
                    fullMessage += " in ";
                else
                    fullMessage += " In ";
            }
        }

        if (hasLocation)
            fullMessage += file + ":" + function + "():" + std::to_string(line);

        else if (!hasMessage) // else assumed !hasLocation
            fullMessage += "[Undefined]";

        return fullMessage;
    }

    std::string getTime()
    {
        // Ubuntu version
        std::time_t rawtime;
        struct std::tm timeStruct;
        std::time(&rawtime);
        timeStruct = *localtime(&rawtime);

        // // Windows version
        // struct std::tm timeStruct;
        // std::time_t time_create{std::time(NULL)};
        // localtime_s(&timeStruct, &time_create);

        // Common
        timeStruct.tm_mon++;
        timeStruct.tm_year += 1900;
        return std::to_string(timeStruct.tm_year) + '_' + std::to_string(timeStruct.tm_mon)
               + '_' + std::to_string(timeStruct.tm_mday) + "___" + std::to_string(timeStruct.tm_hour)
               + '_' + std::to_string(timeStruct.tm_min) + '_' + std::to_string(timeStruct.tm_sec);
    }

    void fileLogging(const std::string& message)
    {
        std::string fileToOpen{"errorLogging.txt"};

        // Get current file size
        std::ifstream in{fileToOpen, std::ios::binary | std::ios::ate};
        const auto currentSizeBytes = in.tellg();
        in.close();

        // Continue at the end of the file or delete it and re-write it (according to current file size)
        const auto maxLogSize = 15 * 1024 * 1024; // 15 MB
        std::ofstream loggingFile{
            fileToOpen, (currentSizeBytes < maxLogSize ? std::ios_base::app : std::ios_base::trunc)};

        // Message to write
        loggingFile << getTime();
        loggingFile << "\n";
        loggingFile << message;
        loggingFile << "\n\n\n\n\n";

        loggingFile.close();
    }

    void errorAux(
        const int errorMode, const std::string& message, const int line, const std::string& function,
        const std::string& file)
    {
        // errorMode:
        // 0: error
        // 1: errorWorker
        // 2: checkWorkerErrors
        // 3: errorDestructor

        const std::string errorInitBase = "\nError";
        const std::string errorInit = errorInitBase + ":\n";
        const std::string errorEnum = "- ";

        // Compose error message
        std::string errorMessageToPropagate;
        std::string errorMessageToPrint;
        // If first error
        if (message.size() < errorInitBase.size() || message.substr(0, errorInitBase.size()) != errorInitBase)
        {
            errorMessageToPrint = errorInit + createFullMessage(message) + "\n\nComing from:\n" + errorEnum
                                + createFullMessage("", line, function, file);
            errorMessageToPropagate = errorMessageToPrint + "\n";
        }
        // If error propagated among different errors
        else
        {
            errorMessageToPrint = errorEnum + createFullMessage("", line, function, file);
            if (errorMode == 2)
            {
                const std::string errorThreadLine =
                    "[All threads closed and control returned to main thread]";
                errorMessageToPrint = errorEnum + errorThreadLine + "\n" + errorMessageToPrint;
            }
            else if (errorMode == 3)
                errorMessageToPrint += "\n" + errorEnum + "[Error occurred in a destructor or in the OpenPose"
                    " Unity Plugin, so no std::exception has been thrown. Returning with exit status 0]";
            errorMessageToPropagate = createFullMessage(message) + errorMessageToPrint + "\n";
            if (errorMode == 1)
            {
                errorMessageToPropagate = errorInitBase + " occurred on a thread. OpenPose closed all its"
                    " threads and then propagated the error to the main thread. Error description:\n\n"
                    + errorMessageToPropagate.substr(errorInit.size(), errorMessageToPropagate.size()-1);
            }
            if (errorMode == 2)
                errorMessageToPrint = errorMessageToPropagate.substr(0, errorMessageToPropagate.size()-1);
        }

        // std::cerr
        if (checkIfErrorHas(ErrorMode::StdCerr))
            #ifdef NDEBUG
            if (!getIfNotInMainThreadOrEmpty())
            #endif
                std::cerr << errorMessageToPrint << std::endl;

        // File logging
        if (checkIfErrorHas(ErrorMode::FileLogging))
            fileLogging(errorMessageToPrint);

        // std::runtime_error
        if (errorMode == 1)
        {
            sThreadErrors = true;
            std::lock_guard<std::mutex> lock{sMutex};
            sThreadErrorMessages.emplace_back(errorMessageToPropagate);
        }
        else
        {
            // Unity logError
            #ifdef USE_UNITY_SUPPORT
                if (errorMode == 3)
                    UnityDebugger::logError(errorMessageToPropagate);
            #endif

            if (checkIfErrorHas(ErrorMode::StdRuntimeError) && errorMode != 3)
                throw std::runtime_error{errorMessageToPropagate};
        }
    }





    // Public functions
    void setMainThread()
    {
        std::lock_guard<std::mutex> lock{sMutex};
        sMainThreadId = getThreadId();
    }

    std::string getThreadId()
    {
        std::stringstream threadId;
        threadId << std::this_thread::get_id();
        return threadId.str();
    }

    bool getIfInMainThreadOrEmpty()
    {
        std::lock_guard<std::mutex> lock{sMutex};
        return (!sMainThreadId.empty() && sMainThreadId == getThreadId());
    }

    bool getIfNotInMainThreadOrEmpty()
    {
        std::lock_guard<std::mutex> lock{sMutex};
        return (!sMainThreadId.empty() && sMainThreadId != getThreadId());
    }

    void error(const std::string& message, const int line, const std::string& function, const std::string& file)
    {
        errorAux(0, message, line, function, file);
    }

    void checkWorkerErrors()
    {
        if (sThreadErrors)
        {
            std::unique_lock<std::mutex> lock{sMutex};
            std::string fullMessage = sThreadErrorMessages.at(0);
            lock.unlock();
            sThreadErrors = false; // Avoid infinity loop throwing the same error over and over.
            errorAux(2, fullMessage, __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void errorWorker(const std::string& message, const int line, const std::string& function, const std::string& file)
    {
        // If we are 100% sure that we are in main thread, then normal error.
        // Otherwise, worker error
        errorAux((getIfInMainThreadOrEmpty() ? 0 : 1), message, line, function, file);
    }

    void errorDestructor(const std::string& message, const int line, const std::string& function, const std::string& file)
    {
        // If we are 100% sure that we are in main thread, then normal error.
        // Otherwise, worker error
        errorAux(3, message, line, function, file);
    }

    void log(const std::string& message, const Priority priority, const int line, const std::string& function,
             const std::string& file)
    {
        if (priority >= ConfigureLog::getPriorityThreshold())
        {
            const auto infoMessage = createFullMessage(message, line, function, file);

            // std::cout
            if (checkIfLoggingHas(LogMode::StdCout))
                std::cout << infoMessage << std::endl;

            // File logging
            if (checkIfLoggingHas(LogMode::FileLogging))
                fileLogging(infoMessage);

            // Unity log
            #ifdef USE_UNITY_SUPPORT
                UnityDebugger::log(infoMessage);
            #endif
        }
    }





    namespace ConfigureError
    {
        // ConfigureError - Private variables
        // std::vector<ErrorMode> sErrorModes              {ErrorMode::StdRuntimeError};
        std::vector<ErrorMode> sErrorModes              {ErrorMode::StdCerr, ErrorMode::StdRuntimeError};
        std::mutex sErrorModesMutex                     {};

        std::vector<ErrorMode> getErrorModes()
        {
            const std::lock_guard<std::mutex> lock{sErrorModesMutex};
            return sErrorModes;
        }

        void setErrorModes(const std::vector<ErrorMode>& errorModes)
        {
            const std::lock_guard<std::mutex> lock{sErrorModesMutex};
            sErrorModes = errorModes;
        }
    }





    namespace ConfigureLog
    {
        // ConfigureLog - Private variables
        std::atomic<Priority> sPriorityThreshold    {Priority::High};
        // std::atomic<Priority> sPriorityThreshold   {Priority::None};
        std::vector<LogMode> sLoggingModes          {LogMode::StdCout};
        // std::mutex sConfigureLogMutex              {}; // In addition, getLogModes() should return copy (no ref)

        Priority getPriorityThreshold()
        {
            return sPriorityThreshold;
        }

        const std::vector<LogMode>& getLogModes()
        {
            // const std::lock_guard<std::mutex> lock{sConfigureLogMutex};
            return sLoggingModes;
        }

        void setPriorityThreshold(const Priority priorityThreshold)
        {
            sPriorityThreshold = priorityThreshold;
        }

        void setLogModes(const std::vector<LogMode>& loggingModes)
        {
            // const std::lock_guard<std::mutex> lock{sConfigureLogMutex};
            sLoggingModes = loggingModes;
        }
    }
}
