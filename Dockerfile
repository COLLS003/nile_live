# Use an official Python 3.11.2 runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask and NLTK (included in requirements.txt)
RUN pip install Flask nltk
RUN pip install scikit-learn
RUN pip install flask-cors


# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run base.py when the container launches
CMD ["python", "app.py"]
