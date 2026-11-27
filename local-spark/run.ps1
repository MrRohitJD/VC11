$ErrorActionPreference = 'Stop'
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:TEMP = Join-Path $PSScriptRoot '.spark-tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:PYSPARK_PYTHON = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Push-Location $PSScriptRoot
try {
    & $env:PYSPARK_PYTHON (Join-Path $PSScriptRoot 'dataframe_demo.py')
    if ($LASTEXITCODE -ne 0) { throw "Spark exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
