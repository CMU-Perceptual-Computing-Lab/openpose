#include <map>
#include <mutex>
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/utilities/profiler.hpp>

// First, I apologize for the ugliness of the code of this function. Nevertheless, it has been made
// in this way so that it has no computational impact at all if PROFILER_ENABLED is not defined.

namespace op
{
    unsigned long long Profiler::DEFAULT_X = 1000;

    std::chrono::time_point<std::chrono::high_resolution_clock> getTimerInit()
    {
        try
        {
            return std::chrono::high_resolution_clock::now();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::chrono::high_resolution_clock::now();
        }
    }

    double getTimeSeconds(const std::chrono::time_point<std::chrono::high_resolution_clock>& timerInit)
    {
        try
        {
            const auto now = std::chrono::high_resolution_clock::now();
            const auto totalTimeSec = double(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerInit).count() * 1e-9);
            return totalTimeSec;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.;
        }
    }

    void printTime(
        const std::chrono::time_point<std::chrono::high_resolution_clock>& timerInit, const std::string& firstMessage,
        const std::string& secondMessage, const Priority priority)
    {
        try
        {
            const auto message = firstMessage + std::to_string(getTimeSeconds(timerInit)) + secondMessage;
            op::log(message, priority);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    #ifdef PROFILER_ENABLED

        std::map<std::string, std::tuple<double, unsigned long long, std::chrono::high_resolution_clock::time_point>> sProfilerTuple{
            std::map<std::string, std::tuple<double, unsigned long long, std::chrono::high_resolution_clock::time_point>>()
        };
        std::mutex sMutexProfiler{};

        std::string getKey(const int line, const std::string& function, const std::string& file)
        {
            return file + function + std::to_string(line) + getThreadId();
        }

        void printAveragedTimeMsCommon(const double timePast, const unsigned long long timeCounter, const int line,
                                       const std::string& function, const std::string& file)
        {
            const auto stringMessage = std::to_string(   timePast / timeCounter / 1e6   ) + " msec";
            log(stringMessage, Priority::Max, line, function, file);
        }
    #endif

    void Profiler::setDefaultX(const unsigned long long defaultX)
    {
        #ifdef PROFILER_ENABLED
            DEFAULT_X = defaultX;
        #else
            UNUSED(defaultX);
        #endif
    }

    const std::string Profiler::timerInit(const int line, const std::string& function, const std::string& file)
    {
        #ifdef PROFILER_ENABLED
            const auto key = getKey(line, function, file);
            std::unique_lock<std::mutex> lock{sMutexProfiler};
            if (sProfilerTuple.count(key) > 0)
                std::get<2>(sProfilerTuple[key]) = std::chrono::high_resolution_clock::now();
            else
                sProfilerTuple[key] = {std::make_tuple(0., 0ull, std::chrono::high_resolution_clock::now())};
            lock.unlock();
            return key;
        #else
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
            return "";
        #endif
    }

    void Profiler::timerEnd(const std::string& key)
    {
        #ifdef PROFILER_ENABLED
            const std::lock_guard<std::mutex> lock{sMutexProfiler};
            if (sProfilerTuple.count(key) > 0)
            {
                auto tuple = sProfilerTuple[key];
                // Time between init & end
                const auto timeNs = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - std::get<2>(tuple)
                ).count();
                // Accumulate averaged time
                std::get<0>(tuple) += timeNs;
                std::get<1>(tuple)++;

                sProfilerTuple[key] = tuple;
            }
            else
                error("Profiler::timerEnd called with a non-existing key.", __LINE__, __FUNCTION__, __FILE__);
        #else
            UNUSED(key);
        #endif
    }

    void Profiler::printAveragedTimeMsOnIterationX(const std::string& key, const int line, const std::string& function,
                                                   const std::string& file, const unsigned long long x)
    {
        #ifdef PROFILER_ENABLED
            std::unique_lock<std::mutex> lock{sMutexProfiler};
            if (sProfilerTuple.count(key) > 0)
            {
                const auto tuple = sProfilerTuple[key];
                lock.unlock();
                if (std::get<1>(tuple) == x)
                {
                    printAveragedTimeMsCommon(std::get<0>(tuple), std::get<1>(tuple), line, function, file);
                }
            }
            else
                error("Profiler::printAveragedTimeMsOnIterationX called with a non-existing key.",
                      __LINE__, __FUNCTION__, __FILE__);
        #else
            UNUSED(key);
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
            UNUSED(x);
        #endif
    }

    void Profiler::printAveragedTimeMsEveryXIterations(const std::string& key, const int line,
                                                       const std::string& function, const std::string& file,
                                                       const unsigned long long x)
    {
        #ifdef PROFILER_ENABLED
            std::unique_lock<std::mutex> lock{sMutexProfiler};
            if (sProfilerTuple.count(key) > 0)
            {
                const auto tupleElement = sProfilerTuple[key];
                lock.unlock();
                if (std::get<1>(tupleElement) == x)
                {
                    printAveragedTimeMsCommon(std::get<0>(tupleElement), std::get<1>(tupleElement), line, function, file);

                    // Reset
                    const std::lock_guard<std::mutex> lockGuard{sMutexProfiler};
                    auto& tuple = sProfilerTuple[key];
                    std::get<0>(tuple) = 0.;
                    std::get<1>(tuple) = 0;
                }
            }
            else
                error("Profiler::printAveragedTimeMsEveryXIterations called with a non-existing key.",
                      __LINE__, __FUNCTION__, __FILE__);
        #else
            UNUSED(key);
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
            UNUSED(x);
        #endif
    }

    void Profiler::profileGpuMemory(const int line, const std::string& function, const std::string& file)
    {
        #ifdef PROFILER_ENABLED
            // Print line-function-file info
            log("GPU usage.", Priority::Max, line, function, file);

            // GPU info
            const auto nvidiaCommand = std::system("nvidia-smi | grep \"Processes:\"")
                                     | std::system("nvidia-smi | grep \"Process name\"");
            if (nvidiaCommand != 0)
                log("Error on the nvidia-smi header. Please, inform us of this error.", Priority::Max);
            else
            {
                // Print GPU usage or empty otherwise
                const std::string fileName{file};
                const std::string getGpuMemoryCommand{"nvidia-smi | grep \"" + file.substr(0, file.size() - 3) + "\""};
                const auto answer = std::system(getGpuMemoryCommand.c_str());
                if (answer == 256)
                    log("Not used at all.", Priority::Max);
                else if (answer != 0)
                    log("Bash error: " + std::to_string(answer), Priority::Max);
            }
        #else
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
        #endif
    }
}
