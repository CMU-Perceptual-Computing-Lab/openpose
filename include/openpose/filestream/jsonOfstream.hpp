#ifndef OPENPOSE_FILESTREAM_JSON_OFSTREAM_HPP
#define OPENPOSE_FILESTREAM_JSON_OFSTREAM_HPP

#include <fstream> // std::ofstream
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API JsonOfstream
    {
    public:
        explicit JsonOfstream(const std::string& filePath, const bool humanReadable = true);

        ~JsonOfstream();

        void objectOpen();

        void objectClose();

        void arrayOpen();

        void arrayClose();

        void version(const std::string& version);

        void key(const std::string& string);

        template <typename T>
        inline void plainText(const T& value)
        {
            mOfstream << value;
        }

        inline void comma()
        {
            mOfstream << ",";
        }

        void enter();

    private:
        const bool mHumanReadable;
        long long mBracesCounter;
        long long mBracketsCounter;
        std::ofstream mOfstream;

        DELETE_COPY(JsonOfstream);
    };
}

#endif // OPENPOSE_FILESTREAM_JSON_OFSTREAM_HPP
