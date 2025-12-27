### This is spatial database final project.

The reference paper is at [LAQP](https://arxiv.org/abs/2003.02446).

Install [POWER dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

Install [Uber Pickups NYC dataset](https://drive.google.com/file/d/1pdWmns1IamjDkMToYapufxCmtb8kFDp3/view?usp=sharing).

Put the installed file to `./data`.

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

<img width="669" height="747" alt="image" src="https://github.com/user-attachments/assets/00b108ca-35f8-462c-b59d-d4973dcfb45b" />





