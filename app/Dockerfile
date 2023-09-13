# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI code into the container
COPY . .

# Expose the port the FastAPI app will run on
EXPOSE 80

# Start the FastAPI app when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
