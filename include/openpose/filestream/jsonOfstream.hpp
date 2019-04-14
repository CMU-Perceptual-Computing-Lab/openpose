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

        /**
         * Move constructor.
         * It destroys the original JsonOfstream to be moved.
         * @param array JsonOfstream to be moved.
         */
        JsonOfstream(JsonOfstream&& jsonOfstream);

        /**
         * Move assignment.
         * Similar to JsonOfstream(JsonOfstream&& jsonOfstream).
         * @param array JsonOfstream to be moved.
         * @return The resulting JsonOfstream.
         */
        JsonOfstream& operator=(JsonOfstream&& jsonOfstream);

        virtual ~JsonOfstream();

        void objectOpen();

        void objectClose();

        void arrayOpen();

        void arrayClose();

        void version(const std::string& version);

        void key(const std::string& string);

        template <typename T>
        inline void plainText(const T& value)
        {
            *upOfstream << value;
        }

        inline void comma()
        {
            *upOfstream << ",";
        }

        void enter();

    private:
        bool mHumanReadable;
        long long mBracesCounter;
        long long mBracketsCounter;
        std::unique_ptr<std::ofstream> upOfstream; // std::unique_ptr to solve std::move issue in GCC < 5

        DELETE_COPY(JsonOfstream);
    };
}

#endif // OPENPOSE_FILESTREAM_JSON_OFSTREAM_HPP
