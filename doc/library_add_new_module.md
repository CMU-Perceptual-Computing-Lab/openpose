OpenPose Library - Steps to Add a New Module
====================================

## Developping Steps
In order to add a new module, these are the recommended steps in order to develop it:

1. Create a folder with its name in the `experimental/` module, e.g. `experimental/hair/`.
2. Implement all the functionality in one `Worker` (i.e. inherit from `Worker` and implement all the functionality on that class).
    1. The first letter of the class name should be `W` (e.g. `WHairExtractor`).
    2. To initially simplify development:
        1. Initialize the Worker class with the specific std::shared_ptr<std::vector<op::Datum>> instead of directly using a template class.
        2. Use the whole op::Datum as unique argument of your auxiliary functions.
        3. Use the OpenPose Wrapper in ThreadManagerMode::SingleThread mode (e.g. it allows you to directly use cv::imshow).
        4. If you are using your own custom Caffe -> initially change the Caffe for your version. It should directly work.
    3. Copy the design from `pose/WPoseExtractor`.
3. To test it:
    1. Add the functionality to `Wrapper`, use the `experimental` namespace for the new Struct (e.g. `experimental::HairStruct`) that the `Wrapper` will use. Do not change any function name from `Wrapper`, just add a new `configure`, with the new `HairStruct` or modify the existing ones without changing their names.
    2. Add a demo (e.g. `examples/openpose/rthair.cpp`) to test it.
4. Split the `Worker` into as many Workers as required.
5. If the Workers need extra data from `Datum`, simply add into `Datum` the new variables required (without removing/modifying any previous variables!).
6. Read also the release steps before starting this developping phase.

## Release Steps
In order to release the new module:

1. Move the functionality of each `Worker` class to the non-template class (e.g. `WHairExtractor` to `HairExtractor`). `WHairExtractor` will simply wrap `HairExtractor`. This will reduce compiling time for the user. See examples from other modules.
2. If you are using a custom Caffe version, move the custom code into the OpenPose library and change back Caffe to the default (most updated) version.
3. Move the module from `experimental/hair/` to `hair/`.
4. Remove `experimental` namespaces (from `Wrapper` and `hair`) and turn Workers into template classes.
5. Add a demo in `examples/openpose/` and tutorial examples in `examples/tutorial_`.
