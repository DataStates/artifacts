#!/usr/bin/env bash
command -v spack

BASE_DIR="$(pwd)"

if [ $? -ne 0 ]; then
  echo "please configure spack"
  exit 1
fi

spack env activate .
spack install

if [ -d ./venv/bin ]; then
  echo "venv already created"
else
  python -m venv venv
fi
source ./venv/bin/activate

python -c "import tensorflow as tf"

if [ $? -ne 0 ]; then
  python -m pip install -r ./requirements.txt
fi

cd "$BASE_DIR/tmci"
python setup.py install

cd "$BASE_DIR"
if [ ! -d cpp-store/build ]; then
  cd cpp-store
  cmake -S . -B ./build -D Python_EXECUTABLE=$(which python)
  cmake --build build
  cd ..
fi
