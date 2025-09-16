git clone --filter=blob:none --no-checkout https://github.com/google-research/google-research.git

cd google-research
git sparse-checkout init --cone
git sparse-checkout set multimodalchat

git checkout

cd ..
python merge_data.py