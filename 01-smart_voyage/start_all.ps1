# 智能旅行系统 一键启动全部服务
$venvPath = ".\venv\Scripts\Activate.ps1"
$workDir = "D:\my_work\01-smart_voyage"

# 启动网关 gateway
Start-Process powershell -ArgumentList "-Command", "cd $workDir; .\$venvPath; python gateway/gateway.py"

# 启动天气智能体 weather_agent
Start-Process powershell -ArgumentList "-Command", "cd $workDir; .\$venvPath; python agents/weather_agent/main.py"

# 启动路线智能体 route_agent
Start-Process powershell -ArgumentList "-Command", "cd $workDir; .\$venvPath; python agents/route_agent/main.py"

# 启动景点智能体 attraction_agent
Start-Process powershell -ArgumentList "-Command", "cd $workDir; .\$venvPath; python agents/attraction_agent/main.py"

Write-Host "✅ 所有服务窗口正在打开！等待10秒后自动打开前端页面..."
Start-Sleep -Seconds 10
Start-Process "file://$workDir\customer.html"