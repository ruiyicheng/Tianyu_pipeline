# Monitor the resource usage of python processes
trap 'echo logged off at $(date +"%Y-%m-%dT%H:%M:%S%z"); exit;' INT
echo "Starting monitor: PID=$$"
echo "PID,cpu,memory,kB_rd/s,kB_wr/s,kB_ccwr/s,cmdname,time" > "monitor$$.csv"
while true
    do
    cpu_and_mem=$(top -n 1 -b | sed '1,7d' | sed 's/^  *//g' | sed 's/  */ /g' | cut -f1,9,10-12 -d' ')
    mysql_cpu=$(echo "$cpu_and_mem"| grep 'mysqld')
    python_cpu=$(echo "$cpu_and_mem"| grep 'python')
    all_cpu_mem="$mysql_cpu"$'\n'$"$python_cpu"
    python_pid=$(echo "$all_cpu_mem" | cut -d' ' -f1 | tr '\n' ',')
    #echo $python_pid
    time_now=$(date +"%Y-%m-%dT%H:%M:%S")
    pydisk="$(sudo pidstat -p "$python_pid" -h -r -u -d 1 1 |  sed '1,3d' | sed 's/  */ /g' | cut -d' ' -f4,9,15-18,20 | tr ' ' ',')"
    #echo "$pydisk"
    
    # duplicate time_now to the same lines as pydisk
    num_lines=$(echo "$pydisk" | wc -l)
    time_now=$(yes "$time_now" | head -n "$num_lines")
    paste <(echo "$pydisk") <(echo "$time_now") -d',' | tr -d "^ "  >> "monitor$$.csv"
done
# mysqld_pid=
