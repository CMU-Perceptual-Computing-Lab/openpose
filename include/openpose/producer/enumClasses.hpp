#ifndef OPENPOSE_PRODUCER_ENUM_CLASSES_HPP
#define OPENPOSE_PRODUCER_ENUM_CLASSES_HPP

namespace op
{
    enum class ProducerFpsMode : bool
    {
        OriginalFps,        /**< The frames will be extracted at the original source fps (frames might be skipped or repeated). */
        RetrievalFps,       /**< The frames will be extracted when the software retrieves them (frames will not be skipped or repeated). */
    };

    enum class ProducerProperty : unsigned char
    {
        AutoRepeat = 0,
        Flip,
        Rotation,
        Size,
    };

    /**
     * Type of producers
     * An enum class in which all the possible type of Producer are included. In order to add a new Producer,
     * include its name in this enum and add a new 'else if' statement inside ProducerFactory::createProducer().
     */
    enum class ProducerType : unsigned char
    {
        FlirCamera,         /**< Stereo FLIR (Point-Grey) camera reader. Based on Spinnaker SDK. */
        ImageDirectory,     /**< An image directory reader. It is able to read images on a folder with a interface similar to the OpenCV cv::VideoCapture. */
        IPCamera,           /**< An IP camera frames extractor, extending the functionality of cv::VideoCapture. */
        Video,              /**< A video frames extractor, extending the functionality of cv::VideoCapture. */
        Webcam,             /**< A webcam frames extractor, extending the functionality of cv::VideoCapture. */
        None,               /**< No type defined. Default state when no specific Producer has been picked yet. */
    };
}

#endif // OPENPOSE_PRODUCER_ENUM_CLASSES_HPP
