FROM python:3.9.15-slim-buster

# set a directory for the app
WORKDIR /home

# copy all the files to the container
COPY ./requirements.txt ./

# install dependencies
RUN \
pip install -r requirements.txt

# define the port number the container should expose
EXPOSE 8000

# run the command
CMD ["python", "src/api.py"]