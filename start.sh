

# Create Alice, Bob, Charlie, Dave, Eve, Ferdie
# echo "Creating wallets for Alice, Bob, Charlie, Dave, Eve, and Ferdie ..."
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Alice'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Alice', overwrite=True, use_password = False, suppress = False); print(w)"
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Bob'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Bob', overwrite=True, use_password = False, suppress = False); print(w)"
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Charlie'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Charlie', overwrite=True, use_password = False, suppress = False); print(w)"
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Dave'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Dave', overwrite=True, use_password = False, suppress = False); print(w)"
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Eve'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Eve', overwrite=True, use_password = False, suppress = False); print(w)"
# python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Ferdie'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_coldkey_from_uri( '/Ferdie', overwrite=True, use_password = False, suppress = False); print(w)"

# Close down all previous processes and restart them.
pm2 sendSignal SIGINT all
pm2 delete all


# python3 master.py --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:0 --use_wandb --min_accumulations 3 --restart --model_type llama
# python3 validator.py --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:1 --use_wandb
# python3 miner.py --wallet.name Alice --wallet.hotkey Bob --subtensor.network test --device cuda:2 --use_wandb
# python3 miner.py --wallet.name Alice --wallet.hotkey Charlie --subtensor.network test --device cuda:3 --use_wandb
# python3 miner.py --wallet.name Alice --wallet.hotkey Dave --subtensor.network test --device cuda:4 --use_wandb
# python3 miner.py --wallet.name Alice --wallet.hotkey Eve --subtensor.network test --device cuda:5 --use_wandb
# python3 miner.py --wallet.name Alice --wallet.hotkey Ferdie --subtensor.network test --device cuda:6 --use_wandb
# Start all the processes again.
pm2 start master.py --interpreter python3 --name Master -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:0 --use_wandb --min_accumulations 3 --restart --model_type llama
pm2 start validator.py --interpreter python3 --name Validator -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:1 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner1 -- --wallet.name Alice --wallet.hotkey Bob --subtensor.network test --device cuda:2 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner2 -- --wallet.name Alice --wallet.hotkey Charlie --subtensor.network test --device cuda:3 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner3 -- --wallet.name Alice --wallet.hotkey Dave --subtensor.network test --device cuda:4 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner4 -- --wallet.name Alice --wallet.hotkey Eve --subtensor.network test --device cuda:5 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner5 -- --wallet.name Alice --wallet.hotkey Ferdie --subtensor.network test --device cuda:6 --use_wandb
# pm2 start baseline.py --interpreter python3 --name Baseline -- --device cuda:7 --use-wandb

