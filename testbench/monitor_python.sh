# Monitor the resource usage of python processes
trap 'echo logged off at $(date +"%Y-%m-%dT%H:%M:%S%z"); exit;' INT
echo "PID,cpu,memory,time,process_name,disk_read_kb,disk_write_kb,time" > "monitor$$.csv"
while true
    do
    cpu_and_mem=$(top -n 1 -b | sed '1,7d' | sed 's/^  *//g' | sed 's/  */ /g' | cut -f1,9,10-12 -d' ')
    mysql_cpu=$(echo "$cpu_and_mem"| grep 'mysqld')
    python_cpu=$(echo "$cpu_and_mem"| grep 'python')
    all_cpu_mem="$mysql_cpu"$'\n'$"$python_cpu"
    while IFS= read -r line; do
        #echo "Processing line: $line"
        # Add your processing logic here
        python_pid=$(echo "$line" | cut -d' ' -f1)
        #echo $python_pid
        time_now=$(date +"%Y-%m-%dT%H:%M:%S")
        pydisk="$(sudo iotop -b -p "$python_pid" -n 1 -k -P -qqq | sed 's/  */ /g' | sed 's/^ *//g'| cut -d' ' -f4,6)"
        #echo $pydisk
        final_line=$(echo "$line $pydisk")
        echo $final_line $time_now | sed 's/ /,/g' >> "monitor$$.csv"
    done <<< "$all_cpu_mem"
done
# mysqld_pid=
