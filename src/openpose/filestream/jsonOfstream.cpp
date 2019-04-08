#include <openpose/filestream/jsonOfstream.hpp>

namespace op
{
    void enterAndTab(std::ofstream& ofstream, const bool humanReadable, const long long bracesCounter,
                     const long long bracketsCounter)
    {
        try
        {
            if (humanReadable)
            {
                ofstream << "\n";
                for (auto i = 0ll ; i < bracesCounter + bracketsCounter ; i++)
                    ofstream << "\t";
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    JsonOfstream::JsonOfstream(const std::string& filePath, const bool humanReadable) :
        mHumanReadable{humanReadable},
        mBracesCounter{0},
        mBracketsCounter{0},
        upOfstream{new std::ofstream{filePath}}
    {
        try
        {
            if (!filePath.empty() && !upOfstream->is_open())
                error("Json file " + filePath + " could not be opened.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    JsonOfstream::JsonOfstream(JsonOfstream&& jsonOfstream) :
        mHumanReadable{jsonOfstream.mHumanReadable},
        mBracesCounter{jsonOfstream.mBracesCounter},
        mBracketsCounter{jsonOfstream.mBracketsCounter}
    {
        try
        {
            upOfstream = std::move(jsonOfstream.upOfstream);
            // std::swap(upOfstream, jsonOfstream.upOfstream);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    JsonOfstream& JsonOfstream::operator=(JsonOfstream&& jsonOfstream)
    {
        try
        {
            mHumanReadable = jsonOfstream.mHumanReadable;
            mBracesCounter = jsonOfstream.mBracesCounter;
            mBracketsCounter = jsonOfstream.mBracketsCounter;
            upOfstream = std::move(jsonOfstream.upOfstream);
            // std::swap(upOfstream, jsonOfstream.upOfstream);
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    JsonOfstream::~JsonOfstream()
    {
        try
        {
            // Moved(std::unique_ptr) will be a nullptr in the old one
            if (upOfstream != nullptr)
            {
                enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);

                if (mBracesCounter != 0 || mBracketsCounter != 0)
                {
                    std::string errorMessage = "Json file wronly generated";
                    if (mBracesCounter != 0)
                        errorMessage += ", number \"{\" != number \"}\": " + std::to_string(mBracesCounter) + ".";
                    else if (mBracketsCounter != 0)
                        errorMessage += ", number \"[\" != number \"]\": " + std::to_string(mBracketsCounter) + ".";
                    else
                        errorMessage += ".";
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                }
            }
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::objectOpen()
    {
        try
        {
            mBracesCounter++;
            *upOfstream << "{";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::objectClose()
    {
        try
        {
            mBracesCounter--;
            enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);
            *upOfstream << "}";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::arrayOpen()
    {
        try
        {
            mBracketsCounter++;
            *upOfstream << "[";
            enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::arrayClose()
    {
        try
        {
            mBracketsCounter--;
            enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);
            *upOfstream << "]";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::version(const std::string& version)
    {
        try
        {
            key("version");
            plainText(version);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::key(const std::string& string)
    {
        try
        {
            enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);
            *upOfstream << "\"" + string + "\":";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void JsonOfstream::enter()
    {
        try
        {
            enterAndTab(*upOfstream, mHumanReadable, mBracesCounter, mBracketsCounter);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
