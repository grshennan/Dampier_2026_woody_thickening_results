# p98_cover.py maps (direct RF p98 + residual maps), detached via WMI (CreateFlags 0x210).
# Log: class_defs\p98_cover_maps.log. Existing maps are skipped, so a relaunch resumes.
#   powershell -File G:\python\dampier_veg_timeseries\class_defs\p98_cover_maps.ps1
param([int]$NJobs = 6)
$cd = 'G:\python\dampier_veg_timeseries\class_defs'
$alive = Get-CimInstance Win32_Process | ? { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'p98_cover\.py' }
if ($alive) { "already running (pid $($alive.ProcessId -join ', ')): not launched"; exit 1 }
$py = 'C:\Users\glens\miniconda3\Scripts\conda.exe run -n general --no-capture-output python -u'
$c = "$py p98_cover.py maps --n-jobs $NJobs >> $cd\p98_cover_maps.log 2>&1"
$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x210; ShowWindow = [uint16]0}
$r = Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine = "cmd.exe /c `"$c`""; CurrentDirectory = $cd; ProcessStartupInformation = $su}
"p98 maps -> return $($r.ReturnValue), pid $($r.ProcessId)"
