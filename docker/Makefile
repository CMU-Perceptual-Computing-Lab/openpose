all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build_cpu            - build cpu image"
	@echo "   1. make build_gpu            - build gpu image"
	# @echo "   1. make pull             - pull all images"
	# @echo "   1. make clean            - remove all images"
	@echo ""

build_cpu:
		cd .. && \
		docker build --tag=cmupcl/openpose:cpu --file=docker/cpu/Dockerfile .

build_gpu:
		cd .. && \
		docker build --tag=cmupcl/openpose:gpu --file=docker/gpu/Dockerfile .

# pull:
# 	@docker pull cmupcl/openpose:gpu
# 	@docker pull cmupcl/openpose:cpu

# clean:
# 	@docker rmi -f cmupcl/openpose:gpu
# 	@docker rmi -f cmupcl/openpose:cpu
