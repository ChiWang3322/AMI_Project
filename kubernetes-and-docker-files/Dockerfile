FROM python:3.8
EXPOSE 8888
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run Web/Homepage.py --server.port 8888
