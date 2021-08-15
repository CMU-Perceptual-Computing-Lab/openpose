#include <openpose/filestream/udpJsonSender.hpp>

#include <iostream>
#include <asio.hpp>

#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/jsonOfstream.hpp>
#include <openpose/utilities/fastMath.hpp>

namespace op
{
	class UdpJsonClient
	{
	public:
		UdpJsonClient(const std::string& host, const std::string& port) :
			mIoService{},
			mUdpSocket{ mIoService, asio::ip::udp::endpoint(asio::ip::udp::v4(), 0) }
		{
			try
			{
				asio::ip::udp::resolver resolver{ mIoService };
				asio::ip::udp::resolver::query query{ asio::ip::udp::v4(), host, port };
				asio::ip::udp::resolver::iterator iter = resolver.resolve(query);
				mUdpEndpoint = *iter;
			}
			catch (const std::exception& e)
			{
				error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}

		~UdpJsonClient()
		{
			try
			{
				mUdpSocket.close();
			}
			catch (const std::exception& e)
			{
				errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}

		void send(const std::string& msg)
		{
			try
			{
				mUdpSocket.send_to(asio::buffer(msg, msg.size()), mUdpEndpoint);
			}
			catch (const std::exception& e)
			{
				error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}
		}

	private:
		asio::io_service mIoService;
		asio::ip::udp::socket mUdpSocket;
		asio::ip::udp::endpoint mUdpEndpoint;
	};

	struct UdpJsonSender::ImplJsonUdpSender
	{
		// Used when increasing spCaffeNets
		UdpJsonClient mUdpClient;

		ImplJsonUdpSender(const std::string& udpHost, const std::string& udpPort) :
			mUdpClient(udpHost, udpPort)
		{
		}
	};

	UdpJsonSender::UdpJsonSender(const std::string& udpHost, const std::string& udpPort)
		: spImpl{ new ImplJsonUdpSender{udpHost, udpPort} }

	{
		// Debugging log
		opLogIfDebug("Starting UDP JSON Sender", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	}

	UdpJsonSender::~UdpJsonSender()
	{
	}

	void UdpJsonSender::sendJson(const std::vector<std::pair<Array<float>, std::string>>& keypointVector)
	{
		try
		{
			// Sanity check
			for (const auto& keypointPair : keypointVector)
				if (!keypointPair.first.empty() && keypointPair.first.getNumberDimensions() != 3
					&& keypointPair.first.getNumberDimensions() != 1)
					error("keypointVector.getNumberDimensions() != 1 && != 3.", __LINE__, __FUNCTION__, __FILE__);

			//Instead of streaming to a file, use a similar means to stream into a string that will be sent over udp
			std::stringbuf streamBuf;
			std::ostream os(&streamBuf);

			//Build the json output - adapted from the file output model
			os << "{";
			try
			{
				// Sanity check
				for (const auto& keypointPair : keypointVector)
					if (!keypointPair.first.empty() && keypointPair.first.getNumberDimensions() != 3
						&& keypointPair.first.getNumberDimensions() != 1)
						error("keypointVector.getNumberDimensions() != 1 && != 3.", __LINE__, __FUNCTION__, __FILE__);
				// Add people keypoints
				os << "\"people\" : ";
				os << "[";
				// Get max numberPeople
				auto numberPeople = 0;
				for (auto vectorIndex = 0u; vectorIndex < keypointVector.size(); vectorIndex++)
					numberPeople = fastMax(numberPeople, keypointVector[vectorIndex].first.getSize(0));
				for (auto person = 0; person < numberPeople; person++)
				{
					os << "{";
					//jsonOfstream.objectOpen();
					for (auto vectorIndex = 0u; vectorIndex < keypointVector.size(); vectorIndex++)
					{
						const auto& keypoints = keypointVector[vectorIndex].first;
						const auto& keypointName = keypointVector[vectorIndex].second;
						const auto numberElementsPerRaw = keypoints.getSize(1) * keypoints.getSize(2);
						os << "\"" + keypointName + "\" : ";
						//Only the person_id node isn't an array (open)
						if (keypointName != "person_id") {
							os << "[";
						}
						// Body parts
						if (numberElementsPerRaw > 0)
						{
							const auto finalIndex = person * numberElementsPerRaw;
							for (auto element = 0; element < numberElementsPerRaw - 1; element++)
							{
								os << keypoints[finalIndex + element];
								os << ",";
							}
							// Last element (no comma)
							os << keypoints[finalIndex + numberElementsPerRaw - 1];
						}

						//Only the person_id node isn't an array (close)
						if (keypointName != "person_id") {
							// Close array
							os << "]";
						}
						if (vectorIndex < keypointVector.size() - 1) {
							os << ",";
						}
					}
					os << "}";
					if (person < numberPeople - 1)
					{
						os << ",";
					}
				}
				// Close bodies array
				os << "]";
				os << "}";
			}
			catch (const std::exception& e)
			{
				error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			}

			//Send the constructed json over UDP
			spImpl->mUdpClient.send(streamBuf.str());

		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
}
