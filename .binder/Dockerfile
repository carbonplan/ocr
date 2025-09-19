FROM ghcr.io/prefix-dev/pixi:latest AS build

# Copy the entire repository (build context) into /app
# Note: build with context at repo root, e.g.:
#   docker build -f .binder/Dockerfile .
COPY . /app
WORKDIR /app

RUN pixi install --environment deploy


# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e deploy > /shell-hook.sh

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh


FROM ubuntu:24.04 AS production

# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/deploy /app/.pixi/envs/deploy
COPY --from=build /shell-hook.sh /shell-hook.sh

# also copy the package source so editable install works in production
# the editable install inside the env points to /app, so we must provide the source there
COPY --from=build /app/ocr /app/ocr

# set a working directory for convenience
WORKDIR /app

# set the entrypoint to the shell-hook script (activate the environment and run the command)
# no more pixi needed in the prod container
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
