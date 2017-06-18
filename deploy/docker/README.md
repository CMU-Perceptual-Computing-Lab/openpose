# Docker

## GPU

Install docker, docker-compose and nvidia-docker, nvidia-docker-compose.

```bash
nvidia-docker-compose build openposegpu
xhost +
nvidia-docker-compose run --rm openposegpu
```