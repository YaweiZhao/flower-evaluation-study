#!/bin/bash
echo "Starting server"
python server.py  --config="/home/yawei/Documents/flower-evaluation-study/config/server-DenseNetConfig.json" &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    python client.py --config="/home/yawei/Documents/flower-evaluation-study/config/client$i-DenseNetConfig.json" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
