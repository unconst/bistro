
 ______ _____    _    _______ ______   _____  
(____  (_____)  | |  (_______|_____ \ / ___ \ 
 ____)  ) _      \ \  _       _____) ) |   | |
|  __  ( | |      \ \| |     (_____ (| |   | |
| |__)  )| |_ _____) ) |_____      | | |___| |
|______(_____|______/ \______)     |_|\_____/ 

---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

# Step 1.
  - Create an S3 <Bucket> on AWS and add export your AWS API Key.
  - export AWS_SECRET_ACCESS_KEY=
  - export AWS_ACCESS_KEY_ID=

# Step 2.
  - Install python3 requirements.
  - `python3 -m pip install -r requirements.txt`

# Step 3. 
  - Register your miner on subnet 212 on testnet.
  - btcli s register --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212

# Step 4.
  - Run your miner.
  - `python3 miner.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket>

