#include <openpose/producer/headers.hpp>

namespace op
{
    template class OP_API DatumProducer<DATUM_BASE_NO_PTR>;
    template class OP_API WDatumProducer<DATUM_BASE, DATUM_BASE_NO_PTR>;
}
