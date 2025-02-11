# Llasa-1B-Multilingual-Test
Llasa-1B-Multilingual

# demo audio

gen.wav

# setup

**build docker image**
```bash
docker build -t llasa-1b-multilingual .
```

**run docker container**
```bash
docker run --gpus all -it -e token -v ${PWD}:/workspace llasa python llasa_sample.py
```
