\begin{itemize}
    \item Table name: \textit{process\_list}

    This table contains the records of processes. 

    Schema:
    \begin{itemize}
        \item \textit{process\_id}: A 25-digit decimal which contains the 19-digit time stamp to nanosecond and a 6-digit. It is generated when the process is being published. This the the foreign key of multiple tables which record the results of the pipeline.
        \item \textit{process\_cmd}: Text with the command to execute combined with the input parameters written in JSON format. Process consumer would decode it for mission execution.
        \item \textit{process\_status\_id}: Status of the process, which is one of waiting for dependence, waiting in queue, executing, successfully executed, and unsuccessfully executed.
        \item \textit{process\_site\_id} \& \textit{process\_group\_id}: Site id and group id of the process to be executed. Consumers at (\textit{process\_site\_id}, \textit{process\_group\_id}) use a unique RabbitMQ channel.
    \end{itemize}

    \item Table name: \textit{img}

    This table contains the metadata of images.

    Schema:
    \begin{itemize}
        \item \textit{image\_id}: The auto-increment id for images
    \end{itemize}

    
    
\end{itemize}