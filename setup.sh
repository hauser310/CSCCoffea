ENV_DIR=.venv

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $ENV_DIR
    source $ENV_DIR/bin/activate
    python3 -m pip install -U pip
    python3 -m pip install -e .[dev]
    pre-commit install
    deactivate
    echo "Creating virtual environment...done"
fi

source activate $ENV_DIR
