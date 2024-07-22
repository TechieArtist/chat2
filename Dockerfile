# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set noninteractive to avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Update packages and install some common utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local files into the container
COPY . .

# Install Python dependencies from requirements.txt
# Make sure your requirements file is in the context directory and adjust the file name if necessary
RUN python3 -m pip install --no-cache-dir -r requirements/pt2.txt

# Set a default command or entrypoint (optional)
CMD ["bash"]
