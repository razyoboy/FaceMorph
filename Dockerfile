FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True

# Copy the current directory contents into the container at /src
COPY . /src

# Set the working directory to /src
WORKDIR /src

# Install build dependencies for dlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available to the world outside this container
EXPOSE $PORT

WORKDIR /src/src
# Run app.py when the container launches
CMD ["python3", "-u", "app.py"]
