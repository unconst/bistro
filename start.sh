

pm2 sendSignal SIGINT all
pm2 delete all
pm2 start master.py --interpreter python3 --name Master -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:0 --use_wandb --n_accumulations 3 --restart --model_type llama
pm2 start validator.py --interpreter python3 --name Validator -- --wallet.name Alice --wallet.hotkey Alice --subtensor.network test --device cuda:1 --use_wandb --n_accumulations 3
pm2 start miner.py --interpreter python3 --name Miner1 -- --wallet.name Alice --wallet.hotkey Bob --subtensor.network test --device cuda:2 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner2 -- --wallet.name Alice --wallet.hotkey Charlie --subtensor.network test --device cuda:3 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner3 -- --wallet.name Alice --wallet.hotkey Dave --subtensor.network test --device cuda:4 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner4 -- --wallet.name Alice --wallet.hotkey Eve --subtensor.network test --device cuda:5 --use_wandb
pm2 start miner.py --interpreter python3 --name Miner5 -- --wallet.name Alice --wallet.hotkey Ferdie --subtensor.network test --device cuda:6 --use_wandb
pm2 start baseline.py --interpreter python3 --name Baselin -- --device cuda:7 --use_wandb