#include "openpose/producer/producerHeaders.hpp"

namespace op
{
	template class DatumProducer<DATUM_BASE_NO_PTR>;
    template class WDatumProducer<DATUM_BASE, DATUM_BASE_NO_PTR>;
}
