 # Instructions
# 1. git clone the repo
# 2. cd HyperscaleES
# 3. replace USERNAME in this Dockerfile
# 4. docker build -t ${USER}_hyperscalees .
# 5. docker run -it --rm -v $(pwd):/app --name ${USER}_yourcontainername ${USER}_hyperscalees python tests/end_to_end_test.py

FROM python:3.11

# Build arguments
ARG USERNAME=aletcher
ARG UID
ARG GID

WORKDIR /app

# Copy the project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install uv for faster package management
RUN pip3 install uv

# Install the package and its dependencies in editable mode
RUN uv pip install -e . --system

# Add user to container
RUN groupadd -f $USERNAME || true
RUN useradd -m -g $USERNAME $USERNAME || useradd -m $USERNAME
RUN chown -R $USERNAME:$USERNAME /app
RUN mkdir -p /app/cache
USER $USERNAME

ENV HF_HUB_CACHE="/app/cache"
ENV HF_HOME="/app/cache"

CMD ["python"]