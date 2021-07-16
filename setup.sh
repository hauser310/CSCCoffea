ENV_DIR=csc.env

echo "Creating virtual environment..."
python -m venv $ENV_DIR
source $ENV_DIR/bin/activate
python -m pip install -U pip
python3 -m pip install -e .[dev]
pre-commit install
echo "Creating virtual environment...done"