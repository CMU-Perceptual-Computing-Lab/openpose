all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build            - build all images"
	# @echo "   1. make pull             - pull all images"
	# @echo "   1. make clean            - remove all images"
	@echo ""

build:
		@docker build --tag=cmupcl/openpose:gpu gpu
		# @docker build --tag=cmupcl/openpose:cpu cpu

# pull:
# 	@docker pull cmupcl/openpose:gpu
# 	@docker pull cmupcl/openpose:cpu

# clean:
# 	@docker rmi -f cmupcl/openpose:gpu
# 	@docker rmi -f cmupcl/openpose:cpu
