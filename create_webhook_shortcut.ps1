$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\TradingView Webhook.lnk")
$Shortcut.TargetPath  = "cmd.exe"
$Shortcut.Arguments   = "/k cd /d `"$PSScriptRoot`" && python webhook_server.py"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.IconLocation = "C:\Windows\System32\SHELL32.dll,13"
$Shortcut.Description  = "TradingView Webhook Server"
$Shortcut.Save()
Write-Host "✅ Shortcut created: Desktop\TradingView Webhook.lnk"
