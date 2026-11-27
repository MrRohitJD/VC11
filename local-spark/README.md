# Local Spark DataFrame exercise

This Python program reads `employees.json` with an explicit schema and runs
Apache Spark locally using two worker threads (`local[2]`). It demonstrates
column selection, filtering, sorting, grouping, and count/sum/average/min/max
aggregations. No cluster or cloud account is needed.

## Run in PowerShell

```powershell
cd D:\Project\VC11\local-spark
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The launcher uses the project's `.venv` and the installed Java 17 at
`C:\Program Files\Java\jdk-17`. Update that path in `run.ps1` if Java moves.
The execution-policy override applies only to this PowerShell process.

## Re-create the Python environment if needed

Requires Python 3.11+ and Java 17+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

PySpark includes the Spark runtime. See the official
[installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).

## Expected results

The filter returns Priya and Meera. Sorting puts Karan first (salary 90000).

| Department | Employees | Total salary | Average salary | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Finance | 3 | 235000 | 78333.33 | 65000 | 90000 |
| HR | 2 | 85000 | 42500.00 | 40000 | 45000 |
| IT | 3 | 205000 | 68333.33 | 60000 | 75000 |

Overall: 8 employees, total salary 525000, average salary 65625.

Spark transformations are lazy: `select`, `filter`, `orderBy`, and `groupBy`
describe the computation. The `show()` actions execute it and print results.
