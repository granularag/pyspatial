# Build
```
docker build -t pyspatial .
```

# Running
 * `-p 8888:8888`  maps internal port 8888 from host to docker
 * `--name` Name of container instance is pyspatial-docker
 * `--rm` remove container after running
 * `jupter_notebook_config.py` should be placed in `/etc/jupyter/`
 * SSL certs should placed in `/etc/jupyter/certs/jupyter.[key, pem]`
 * Use `/notebooks` inside the container for your notebooks 
 * Can use `/data` inside the container to mount your data

```bash
PYSPATIAL_DIR=/path/to/pyspatial
docker run --rm -v $PYSPATIAL_DIR/examples:/notebooks/pyspatial -v ./jupyter:/etc/jupyter -v $PYSPATIAL_DIR/test:/notebooks/test -p 8889:8888 --name pyspatial-docker pyspatial
```
