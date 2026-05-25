# Dockerfile for Cascade Pneumonia Detection
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools as user
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY --chown=user requirements.txt requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Clone mask-rcnn và copy vào thư mục app (không cài như package)
RUN git clone --depth 1 https://github.com/akTwelve/Mask_RCNN.git /tmp/mask-rcnn && \
    cd /tmp/mask-rcnn && \
    git checkout d9f01dcd83b1e90e5ea0b4b6d5ba753770e114ac && \
    cp -r mrcnn /app/ && \
    rm -rf /tmp/mask-rcnn

# Copy application files (requirements.txt already copied above)
COPY --chown=user . /app

# Expose port
EXPOSE 7860

# Run the application with Flask
CMD ["python", "app.py"]
