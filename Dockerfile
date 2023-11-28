FROM python:3.9  

WORKDIR /ml_app

COPY peft_fine_tune_model.py /ml_app/

COPY . .

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["python","peft_fine_tune_model.py"]  # will run python peft_fine_tune_model.py and starts the application, it is the last command to execute