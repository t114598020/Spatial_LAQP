### This is spatial database final project.

The reference paper is [LAQP (Learning-based Approximate Query Processing](https://arxiv.org/abs/2003.02446).

We reproduced LAQP and compared it with SAQP (sampling-based AQP) and AQP++ (pre-aggregate AQP with range similar) on POWER and Uber Pickups NYC dataset, experiments show that LAQP has better performance than SAQP and AQP++, but will cost more time because it used error prediction model, the images below are the ARE (Average Relative Error) result evaluated on POWER and Uber Pickups NYC dataset.

<img src="https://github.com/t114598020/Spatial_LAQP/blob/main/images/power_are_result?raw=true" height="625px" width="500px" />
<img src="https://github.com/t114598020/Spatial_LAQP/blob/main/images/uber_are_result?raw=true" height="625px" width="500px" />


### Data

Our dataset is Uber Pickups in New York City, the original website is at [kaggle](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city).

Install [POWER dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) for the LAQP_power.ipynb.

Install pre-processed [Uber Pickups NYC dataset](https://drive.google.com/file/d/1pdWmns1IamjDkMToYapufxCmtb8kFDp3/view?usp=sharing) for the LAQP_uber.ipynb and the demo, we only use uber-raw-data.csv from April to September.

Put the installed file to `./data/`.

### Environment

Python 3.10.19

Run `pip install -r requirements.txt` for the environment.

To run demo, `streamlit run demo.py`.

### Weights

Give pre-trained model's weight for uber and power dataset in `./weights`.

### Demo

Demo will build a streamlit website on localhost.

User can choose city like Brooklyn, Manhattan and strict the latitude and longitude.

There is a calender which bound Date/Time according to the Uber Pickups dataset.

Also, user can draw a bbox to customize query.

<img src="https://github.com/t114598020/Spatial_LAQP/blob/main/images/demo_query.png?raw=true" height="625px" width="500px" />
<img src="https://github.com/t114598020/Spatial_LAQP/blob/main/images/demo_draw.png?raw=true" height="630px" width="500px" />
<img src="https://github.com/t114598020/Spatial_LAQP/blob/main/images/demo_result.png?raw=true" height="750px" width="500px" />




















