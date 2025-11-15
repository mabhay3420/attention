if [ ! -d .venv ]; then
	uv venv -p 3.10
fi
source .venv/bin/activate
uv pip install -r requirements.txt
python download.py
python model.py
