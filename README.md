# helloMilvus
Search 100k images from VNExpress using Milvus and Redis

### Step 1: Start Milvus server
```
! wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
! docker-compose up -d
```

### Step 2: Start Redis server
```
! docker run --name redis -d -p 6379:6379 redis
```

### Step 3: Import data
```
! python import_data.py
```

### Step 4: Run demo server
```
! python server.py
```
