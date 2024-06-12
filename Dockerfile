FROM python:3-slim
LABEL maintainer="jacesca@gmail.com"
WORKDIR /app
RUN mkdir datasets
RUN mkdir local-saved-models
RUN mkdir saved-images
COPY datasets/. datasets/.
COPY *.py .
COPY requirements.txt .
COPY .envsample .env
RUN pip install -r requirements.txt --root-user-action=ignore
EXPOSE 5000
ENV GIT_PYTHON_REFRESH=quiet
CMD ["python", "end-to-end.py"]

# $ docker build -t heart_disease_model .
# $ docker run -it --name heart_disease_model -v C:\Users\Jacqueline\Documents\projects\CAMP-MLEngTrack\2-EndToEnd\saved-images:/app/saved-images -p 5000:5000 heart_disease_model