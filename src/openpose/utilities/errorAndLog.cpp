#include <ctime> // std::tm, std::time_t
#include <fstream> // std::ifstream, std::ofstream
#include <iostream> // std::cout, std::endl
#include <stdexcept> // std::runtime_error
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
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
        std::ofstream loggingFile{fileToOpen,
                                  (currentSizeBytes < maxLogSize ? std::ios_base::app : std::ios_base::trunc)};

        // Message to write
        loggingFile << getTime();
        loggingFile << "\n";
        loggingFile << message;
        loggingFile << "\n\n\n\n\n";

        loggingFile.close();
    }





    // Public functions
    void error(const std::string& message, const int line, const std::string& function, const std::string& file)
    {
        const std::string errorInit = "\nError:\n";
        const std::string errorEnum = "- ";

        // Compose error message
        std::string errorMessageToPropagate;
        std::string errorMessageToPrint;
        // If first error
        if (message.size() < errorInit.size() || message.substr(0, errorInit.size()) != errorInit)
        {
            errorMessageToPrint = errorInit + createFullMessage(message) + "\n\nComing from:\n" + errorEnum
                                + createFullMessage("", line, function, file);
            errorMessageToPropagate = errorMessageToPrint + "\n";
        }
        // If error propagated among different errors
        else
        {
            errorMessageToPrint = errorEnum + createFullMessage("", line, function, file);
            errorMessageToPropagate = createFullMessage(message.substr(0, message.size()-1)) + "\n"
                                    + errorMessageToPrint + "\n";
        }

        // std::cerr
        if (checkIfErrorHas(ErrorMode::StdCerr))
            std::cerr << errorMessageToPrint << std::endl;

        // File logging
        if (checkIfErrorHas(ErrorMode::FileLogging))
            fileLogging(errorMessageToPrint);

        // std::runtime_error
        if (checkIfErrorHas(ErrorMode::StdRuntimeError))
            throw std::runtime_error{errorMessageToPropagate};
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
        }
    }





    // ConfigureError - Private variables
    // std::vector<ErrorMode> sErrorModes              {ErrorMode::StdRuntimeError};
    std::vector<ErrorMode> sErrorModes              {ErrorMode::StdCerr, ErrorMode::StdRuntimeError};
    std::mutex sErrorModesMutex                     {};

    std::vector<ErrorMode> ConfigureError::getErrorModes()
    {
        const std::lock_guard<std::mutex> lock{sErrorModesMutex};
        return sErrorModes;
    }

    void ConfigureError::setErrorModes(const std::vector<ErrorMode>& errorModes)
    {
        const std::lock_guard<std::mutex> lock{sErrorModesMutex};
        sErrorModes = errorModes;
    }





    // ConfigureLog - Private variables
    std::atomic<Priority> sPriorityThreshold        {Priority::High};
    // std::atomic<Priority> sPriorityThreshold        {Priority::None};
    std::vector<LogMode> sLoggingModes              {LogMode::StdCout};
    std::mutex sConfigureLogMutex                   {};

    Priority ConfigureLog::getPriorityThreshold()
    {
        return sPriorityThreshold;
    }

    const std::vector<LogMode>& ConfigureLog::getLogModes()
    {
        const std::lock_guard<std::mutex> lock{sConfigureLogMutex};
        return sLoggingModes;
    }

    void ConfigureLog::setPriorityThreshold(const Priority priorityThreshold)
    {
        sPriorityThreshold = priorityThreshold;
    }

    void ConfigureLog::setLogModes(const std::vector<LogMode>& loggingModes)
    {
        const std::lock_guard<std::mutex> lock{sConfigureLogMutex};
        sLoggingModes = loggingModes;
    }
}
