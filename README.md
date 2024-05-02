# dso599-llm-chatbot

## Environment Set up

### Conda virtual environment
Requires conda/Anaconda python distribution for set up
```
conda env create -f environment.yml
```

Afterwards, to activate the virtual environment
```
conda activate dso599_llm_chatbot
```

### Additional setup
Access Keys
* Since this code uses OpenAI API and OpenWeatherMap API, it requires access keys. Modify the `.env` file within `llm_chatbot` folder with your keys.
* The code also requires setting up aws credentials, and assumes set up via aws-cli. However it can be modified to use the aws access keys specified in `.env` by changing the `dynamodb_db.py` code accordingly

## Running stream lit app
To run the streamlit app, navigate to llm_chatbot folder and run
```
streamlit run streamlit_chatbot.py --server.port 8501
```

## Notebook example
Within the notebook folder, `dso599_chatbot_feature_testing.ipynb` contains example code for running each part of the assignment

## Part 4 - End-to-End workflow example
Navigate to llm_chatbot folder and run `python dino_safety_flow_part4.py`
